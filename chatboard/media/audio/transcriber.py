import torch
from components.text.util import sanitize_sentence
import torchaudio
from torchaudio.models.decoder import ctc_decoder, download_pretrained_files
from torchaudio.utils import download_asset
import hashlib
import string
import numpy as np
import re
import os
from config import DEBUG, TEMP_DIR
from termcolor import colored



if DEBUG:
    try:
        import matplotlib.pyplot as plt
    except:
        pass

_RE_COMBINE_WHITESPACE = re.compile(r"\s+")



class GreedyHypothesis():

    def __init__(self, char_hypothesis) -> None:
        words = []
        curr_word = ""
        for c in [h['char'] for h in char_hypothesis]:
            if c == "|":
                words.append(curr_word)
                curr_word = ""
            else:
                curr_word += c
        self.words = words
        self.timesteps = torch.tensor([h['idx'] for h in char_hypothesis])
        self.tokens = [h['char'] for h in char_hypothesis]        
        # self.tokens = [h['token'][1] for h in char_hypothesis]        


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank
        self.label_map = {self.labels[i]:i for i in range(len(self.labels))}

    def idxs_to_tokens(self, tokens):
        return tokens
        # return [self.labels[i] for i in tokens]

    def forward(self, emission: torch.Tensor, sentence: str):
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        san_sentence = sanitize_sentence(sentence)
        text = san_sentence.replace(" ", "|")
        emission = emission[0]
        
        char_idx = 0

        def safe_char(idx):
            if idx < 0:
                return ""
            if idx >= len(text):
                return ""
            return text[idx]
        
        char_hypothesis = []
        hypothesis_history = []
        window_size = 1
        for i in range(emission.shape[0]):
            token_vec = emission[i]
            idx = torch.topk(token_vec, 5).indices
            # best = token_vec[idx]
            token_options = [(self.labels[j],token_vec[j].item()) for j in idx]
            token_options.sort(key=lambda x: x[1], reverse=True)
            token_options_labels = [h[0] for h in token_options]
            

            if safe_char(char_idx) in token_options_labels:
                char_hypothesis.append({'char':text[char_idx], 'idx': i, 'hyp': token_options})
                char_idx += 1
            # if token := check_if_char_in_tokens(safe_char(char_idx), token_options):
            #     char_hypothesis.append({'char':text[char_idx], 'idx': i, 'hyp': token_options, 'token': token})
            #     char_idx += 1
            hypothesis_history.append(token_options)
        print("".join([h['char'] for h in char_hypothesis]))
        print("======")

        greedy_hypothesis = GreedyHypothesis(char_hypothesis=char_hypothesis)
        original_words = san_sentence.split(" ")
        edit_distance = torchaudio.functional.edit_distance(original_words, greedy_hypothesis.words)
        return greedy_hypothesis, edit_distance

        
            # best_char = [(self.labels[j],token_vec[j].item()) for j in idx]
            # best_char.sort(key=lambda x: x[1], reverse=True)
            # print(best_char)
        # indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        # indices = torch.unique_consecutive(indices, dim=-1)
        # indices = [i for i in indices if i != self.blank]
        # joined = "".join([self.labels[i] for i in indices])
        # return joined.replace("|", " ").strip().split()



# def sanitize_sentence(sentence):
#     sentence = re.sub(r'\[[^()]*\]', '', sentence)
#     sentence = re.sub(r'[^\w\s]', '', sentence)
#     sentence = _RE_COMBINE_WHITESPACE.sub(" ", sentence).strip().lower()
#     sentence = sentence.translate(str.maketrans('', '', string.punctuation))
#     return sentence

def get_time(waveform, sample_rate):
    return len(waveform) / sample_rate


def tensor2time(val):
    return val.item()


def plot_alignments(waveform, emission, tokens, timesteps, sample_rate):
    fig, ax = plt.subplots(figsize=(32, 10))

    ax.plot(waveform)

    ratio = waveform.shape[0] / emission.shape[1]
    word_start = 0

    for i in range(len(tokens)):
        if i != 0 and tokens[i - 1] == "|":
            word_start = timesteps[i]
        if tokens[i] != "|":
            plt.annotate(tokens[i].upper(), (timesteps[i] * ratio, waveform.max() * 1.02), size=14)
        elif i != 0:
            word_end = timesteps[i]
            ax.axvspan(word_start * ratio, word_end * ratio, alpha=0.1, color="red") #vertical span rectangle

    xticks = ax.get_xticks()
    plt.xticks(xticks, xticks / sample_rate)
    ax.set_xlabel("time (sec)")
    ax.set_xlim(0, waveform.shape[0])



def merge_transcript_with_ner_sentence(word_timestemps, ner_sentence, threshold=0.25):
    orig_words = ner_sentence.lexicon_ids
    print('merging transcript with ner sentence')
    
    def word_distance(w1, w2):
        curr_treshold = round((len(w1) + len(w2) / 2) * threshold) + 1
        distance = torchaudio.functional.edit_distance(w1.lower(), w2.lower())
        return distance, curr_treshold
        # return torchaudio.functional.edit_distance(w1.lower(), w2.lower()) <= curr_treshold

    def comp_transcript_with_baseline(i_t, i_o, error_type='equality'):
        if i_o < len(orig_words) and i_t < len(word_timestemps):
            distance, curr_treshold = word_distance(word_timestemps[i_t]['word'], orig_words[i_o].word) 
            print(f"   - {colored('√', 'green') if distance <= curr_treshold else 'X'} checking {error_type} {i_t}{colored(' '+word_timestemps[i_t]['word']+' ', 'black', 'on_light_magenta')} ≈ {i_o}{colored(' '+orig_words[i_o].word+' ', 'black', 'on_light_blue')} = {colored(distance, 'green' if distance <= curr_treshold else 'red')} ({'≤' if distance <= curr_treshold else '>'} {curr_treshold})")
            # print('   - checking',error_type,i_t, word_timestemps[i_t]['word'], i_o, orig_words[i_o].word, '(dis', distance, '<', curr_treshold, distance <= curr_treshold,')')
            return distance <= curr_treshold
        else:
            print('our of range', i_t, i_o)
            return False
    
    def set_timestamp(i_t, i_o, set_chars=True):
        ner_word = ner_sentence.get_word_by_id(orig_words[i_o].id)
        ner_word.add_timestamp(
            word_timestemps[i_t]['time'], 
            word_timestemps[i_t]['duration'], 
            word_timestemps[i_t]['chars'] if set_chars else None
        )
        

    def print_error(i_t, i_o, error_type='equality'):
        ner_word = ner_sentence.get_word_by_id(orig_words[i_o].id)
        print(f"{error_type}:[{i_t}][{i_o}] \"{word_timestemps[i_t]['word']}\" = \"{orig_words[i_o].word}\", orig: {ner_word.orig_word}")

    def print_equality(i_t, i_o):
        ner_word = ner_sentence.get_word_by_id(orig_words[i_o].id)
        print(f"{colored('EQUALITY','green')}:[{i_t}][{i_o}] \"{word_timestemps[i_t]['word']}\" = \"{orig_words[i_o].word}\", orig: {ner_word.orig_word}")

    def print_addition(i_t, i_o):
        ner_word = ner_sentence.get_word_by_id(orig_words[i_o].id)
        print(f"{colored('ADDITION', 'red')}: {i_t}{colored(' '+word_timestemps[i_t]['word']+' ','black', 'on_light_magenta')} was added, deleting it, orig: {ner_word.orig_word}")

    def print_deletion(i_o):
        ner_word = ner_sentence.get_word_by_id(orig_words[i_o].id)
        print(f"{colored('DELETION', 'red')}: {i_o}{colored(' ' +orig_words[i_o].word+' ',  'black', 'on_light_blue')} was deleted. adding and fixing time.  orig: {ner_word.orig_word}")

    def print_substitution(i_t, i_o):
        ner_word = ner_sentence.get_word_by_id(orig_words[i_o].id)
        print(f"{colored('SUBSTITUTION','yellow')}: changed {i_o}{colored(' ' +orig_words[i_o].word+' ', 'black', 'on_light_blue')} => {i_t}{colored(' ' +word_timestemps[i_t]['word']+' ', 'black', 'on_light_magenta')}, orig: {ner_word.orig_word}")
        # print(error_type, ':',i_t, word_timestemps[i_t]['word'], i_o, orig_words[i_o].word, ner_word.orig_word)



    print('time alignment')
    print('transcript words:\n', ' '.join([f"{i}{colored(w['word'], 'black', 'on_light_magenta')}" for i,w in enumerate(word_timestemps)]), '.\n')
    print('original words:\n', ' '.join([f"{i}{colored(w.word, 'black', 'on_light_blue')}" for i,w in enumerate(orig_words)]), '.\n')
    

    errors = 0
    fatal_errors = 0

    trans_word = 0
    orig_word = 0
    
    while trans_word < len(word_timestemps):
        # orig_word = 
        if orig_word >= len(orig_words) or trans_word >= len(word_timestemps):
            break        
        #? types of errors: addition, deletion, substitution
        #? assumption that the past words are correct
        if comp_transcript_with_baseline(trans_word, orig_word):
            #? equality
            set_timestamp(trans_word, orig_word)
            print_equality(trans_word, orig_word)
            trans_word += 1
            orig_word += 1
        elif comp_transcript_with_baseline(trans_word + 1, orig_word, 'addition'): #? addition
            #? current word was added.
            print_addition(trans_word, orig_word)
            trans_word += 1
            errors += 1
            fatal_errors += 1
        elif comp_transcript_with_baseline(trans_word, orig_word + 1, 'deletion'):  #? deletion            
            print_deletion(orig_word)
            orig_word += 1
            errors += 1
            fatal_errors += 1
        elif comp_transcript_with_baseline(trans_word + 1, orig_word + 1, 'substitution'): #? substitution
            set_timestamp(trans_word, orig_word, set_chars=False)            
            print_substitution(trans_word, orig_word)
            trans_word += 1
            orig_word += 1
            errors += 1
        else:
            # ? error
            fatal_errors += 1
            errors += 1
            if trans_word < len(word_timestemps):
                print_error(trans_word, orig_word, colored('UNKNOWN ERROR','red') + ' skipping both words.')
            trans_word += 1
            orig_word += 1

    return ner_sentence, errors, fatal_errors


def merge_transcript_with_original(transcript, original_sentence, trashhold=0.3):
    orig_words = original_sentence.split(' ')

    def word_distance(w1, w2):
        curr_treshold = round((len(w1) + len(w2) / 2) * trashhold) + 1
        return torchaudio.functional.edit_distance(w1.lower(), w2.lower()) <= curr_treshold

    i_t = 0
    i_o = 0
    while i_t < len(transcript):
        if i_o >= len(orig_words):
            break

        if word_distance(transcript[i_t]['word'], orig_words[i_o]):
            print('curr word', transcript[i_t]['word'], orig_words[i_o])
            transcript[i_t]['punct_word'] = orig_words[i_o] + ' '
            i_t += 1
            i_o += 1
        elif i_o + 1 <  len(orig_words) and word_distance(transcript[i_t]['word'], orig_words[i_o + 1]):
            print('next word', transcript[i_t]['word'], orig_words[i_o + 1])
            # transcript[i_t]['punct_word'] = orig_words[i_o + 1] + ' '
            i_o += 2
        elif i_t + 1 < len(transcript) and word_distance(transcript[i_t + 1]['word'], orig_words[i_o]):
            print('prev word', transcript[i_t + 1]['word'], orig_words[i_o])
            transcript[i_t + 1]['punct_word'] = orig_words[i_o] + ' '
            i_t += 2
        else:
            print('skiping word', transcript[i_t]['word'], orig_words[i_o])
            i_t += 1
            i_o += 1
    return transcript


class Transcriber:



    def __init__(self, lm_weight=3.23, word_score=-0.26):
        # self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_10M
        # self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_10M
        # self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_10M
        # self.bundle = torchaudio.pipelines.VOXPOPULI_ASR_BASE_10K_EN
        self.bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
        # self.bundle = torchaudio.pipelines.HUBERT_ASR_XLARGE
        self.acoustic_model = self.bundle.get_model()
        files = download_pretrained_files("librispeech-4-gram")
        self.files = files
        # self.beam_search_decoder = ctc_decoder(
        #     lexicon=files.lexicon,
        #     tokens=files.tokens,
        #     lm=files.lm,
        #     nbest=3,
        #     beam_size=1500,
        #     lm_weight=lm_weight,
        #     word_score=word_score,
        # )


    def transcribe_file(self, speech_file):
        waveform, sample_rate = torchaudio.load(speech_file)
        return self.transcribe_waveform(waveform, sample_rate=sample_rate)
    
    # def sentence_to_lexicon(self, sentence):
    #     sentence = sentence.replace("-", "")
    #     sentence = _RE_COMBINE_WHITESPACE.sub(" ", sentence).strip().lower()
    #     sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    #     lexicon = ""
    #     words = list(set(sentence.split(" ")))
    #     words.sort()
        
    #     for word in words:
    #         # sun_word = re.sub(r'[^a-zA-Z0-9]', '', word)
    #         sun_word = re.sub(r'[^a-zA-Z]', '', word)
    #         sun_word = sun_word.replace(" ", "")
    #         if sun_word == "":
    #             continue
    #         chars = list(sun_word)
    #         # lexicon += f"{word}\t{' '.join(chars)} |\n"
    #         lexicon += f"{word} {' '.join(chars)} |\n"
    #     lexicon_filename = str(TEMP_DIR / f"{hashlib.md5(lexicon.encode()).hexdigest()}lexicon.txt")
    #     # lexicon_filename = str(TEMP_DIR / f"lexicon.txt")
    #     with open(lexicon_filename, "w") as f:
    #         f.write(lexicon)
    #     return lexicon_filename, lexicon

    def sentence_to_lexicon(self, sentence):
        
        lexicon, lexicon_ids = sentence.get_lexicon()

        print(lexicon)

        lexicon_filename = str(TEMP_DIR / f"{hashlib.md5(lexicon.encode()).hexdigest()}lexicon.txt")
        # lexicon_filename = str(TEMP_DIR / f"lexicon.txt")
        with open(lexicon_filename, "w") as f:
            f.write(lexicon)
        return lexicon_filename, lexicon
    

    # def get_decoder(self, lexicon=None, lm_weight=3.23, word_score=-0.26):
    def get_decoder(self, use_greedy=False ,lexicon=None, lm_weight=0.23, word_score=-0.26):
        if use_greedy:
            tokens = [label.lower() for label in self.bundle.get_labels()]
            greedy_decoder = GreedyCTCDecoder(tokens)
            return greedy_decoder
        
        beam_search_decoder = ctc_decoder(
            lexicon=lexicon if lexicon else self.files.lexicon,
            tokens=self.files.tokens,
            # lm=self.files.lm,
            lm=None,
            nbest=10,
            beam_size=1500,
            lm_weight=lm_weight,
            word_score=word_score,
            # word_score=-10,
            beam_size_token=len(self.files.tokens)
        )
        return beam_search_decoder
        
        
    

    def choose_hypothesis(self, beam_search_result, original_sentence=None):
        if original_sentence is None:
            return beam_search_result[0][0], -1
        original_sentence = sanitize_sentence(original_sentence).split(" ")
        min_edit_distance = 100000
        min_index = 0
        min_beam_index = 0
        for j in range(len(beam_search_result)):
            for i in range(len(beam_search_result[min_beam_index])):
                hypothesis = beam_search_result[min_beam_index][i]
                # print(hypothesis.words)
                edit_distance = torchaudio.functional.edit_distance(original_sentence, hypothesis.words)
                if edit_distance < min_edit_distance:
                    min_edit_distance, min_index, min_beam_index = edit_distance, i, j

        return beam_search_result[min_beam_index][min_index], min_edit_distance



    def transcribe_waveform(self, waveform, sample_rate, sentence=None, ner_sentence=None, plot_alignment=False, prefix_padding_sec=0, threshold=0.25, verbose=False, use_greedy=False):
        if verbose:
            print('transcribing waveform')
        if type(waveform) == np.ndarray:
            waveform = torch.from_numpy(waveform).float()
            if prefix_padding_sec:
                silence = torch.zeros(int(prefix_padding_sec * sample_rate))
                # padded_waveform = torch.concatenate([silence, waveform[0]], dim=0)
                # waveform = torch.from_numpy(waveform).float()
                waveform = torch.concatenate([silence, waveform], dim=0)
        if sample_rate != self.bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.bundle.sample_rate)

        if waveform.ndim == 1:
            waveform = torch.stack([waveform, waveform], dim=0)

        lexicon = None
        lexicon_filename = None
        if ner_sentence and not use_greedy:
            lexicon_filename, lexicon = self.sentence_to_lexicon(ner_sentence)
            if verbose:
                print('lexicon:\n', lexicon)
        
        # waveform = waveform.type(torch.DoubleTensor)
        beam_search_decoder = self.get_decoder(use_greedy=use_greedy, lexicon=lexicon_filename)

        emission, _ = self.acoustic_model(waveform)

        if use_greedy:
            best_hypothesis, edit_distance = beam_search_decoder(emission, ner_sentence.aug_sentence())
        else:
            beam_search_result = beam_search_decoder(emission)
            best_hypothesis, edit_distance = self.choose_hypothesis(beam_search_result, ner_sentence.aug_sentence())

        timesteps = best_hypothesis.timesteps

        predicted_tokens = beam_search_decoder.idxs_to_tokens(best_hypothesis.tokens)
        #? time steps to seconds
        timesteps_in_seconds = timesteps * ((waveform[0].shape[0] / sample_rate) / emission.shape[1]) - prefix_padding_sec


        #------------------------------
        ratio = waveform[0].shape[0] / emission.shape[1]

        words = self.pack_timesteps(timesteps, predicted_tokens, ratio, self.bundle.sample_rate)

        duration = waveform[0].shape[0]/ self.bundle.sample_rate

        transcript_text = " ".join(best_hypothesis.words)

        if plot_alignment:
            plot_alignments(waveform[0], emission, predicted_tokens, timesteps, sample_rate)
        if lexicon_filename:
            os.remove(lexicon_filename)
        
        if verbose:
            print('original text:\n', ner_sentence.orig_sentence())
            print('augmented text:\n', ner_sentence.aug_sentence())
            print('transcribed text:\n', transcript_text)
        
        # WER = 0
        # if edit_distance:
        #     WER = edit_distance / len(sentence.split(' '))
        #     if verbose:
        #         print('original text:\n', sentence)
        #         print('edit distance:', edit_distance, 'WER:', WER)

        # if sentence:
        #     words = merge_transcript_with_original(words, sentence)
        errors = 0
        fatal_errors = 0
        time_fixes = 0

        if ner_sentence:
            ner_sentence, errors, fatal_errors = merge_transcript_with_ner_sentence(words, ner_sentence, threshold=threshold)
        try:
            time_fixes = ner_sentence.auto_fix_time()
        except Exception as e:
            time_fixes = -1
            print('Error: could not fix time')
            fatal_errors+=1

        try:
            word_duration = ner_sentence.get_word_duration()
        except Exception as e:
            word_duration = duration
            print('Error: could not get word duration. using original duration')
            fatal_errors+=1
        
        return {
            'sentence': transcript_text,
            'start': 0,
            'orig_duration': duration,
            'duration': word_duration,
            'end': duration,
            'words': ner_sentence.to_json(),
            
            # 'WER': WER,            
        }, {
            'edit_distance': edit_distance,
            'errors': errors,
            'fatal_errors': fatal_errors,
            'time_fixes': time_fixes
        }
    
    def print_transcript(self, words):        
        print(" ".join([word["word"] for word in words]))
         


    def pack_timesteps(self, timesteps, predicted_tokens, ratio, sample_rate):
        
        if len(timesteps) != len(predicted_tokens):
            raise ValueError("timesteps and predicted_tokens must have the same length")
        
        def t2s(t):
            return t * ratio / sample_rate
        
        words = []
        curr_word = ""      
        curr_chars = []     
        for i in range(len(timesteps)):
            token = predicted_tokens[i]
            if token == "|":
                if curr_word != "" and len(curr_chars) > 0:
                    # for c_i in range(len(curr_chars) - 1):
                        # curr_chars[c_i]["duration"] = curr_chars[c_i + 1]["time"] - curr_chars[c_i]["time"]
                        # curr_chars[c_i]["duration"] = timesteps[i + c_i + 1]["time"] - timesteps[i + c_i]["time"]
                    words.append({
                        "word": curr_word,
                        "time": curr_chars[0]["time"],
                        "duration": curr_chars[-1]["time"] + curr_chars[-1]["duration"] - curr_chars[0]["time"],
                        "chars": curr_chars,
                    })
                curr_word = ""
                curr_chars = []
            else:
                curr_word += token
                curr_chars.append({
                    "char": token,
                    "time": tensor2time(t2s(timesteps[i])),
                    "duration": tensor2time(t2s(timesteps[i+1]) - t2s(timesteps[i])) if i < len(timesteps) - 1 else 0,
                })
        return words
    
    def pack_timesteps2(self, timesteps, predicted_tokens):
        
        if len(timesteps) != len(predicted_tokens):
            raise ValueError("timesteps and predicted_tokens must have the same length")
        words = []
        curr_word = ""      
        curr_chars = []     
        for i in range(len(timesteps)):
            token = predicted_tokens[i]
            if token == "|":
                if curr_word != "" and len(curr_chars) > 0:
                    # for c_i in range(len(curr_chars) - 1):
                        # curr_chars[c_i]["duration"] = curr_chars[c_i + 1]["time"] - curr_chars[c_i]["time"]
                        # curr_chars[c_i]["duration"] = timesteps[i + c_i + 1]["time"] - timesteps[i + c_i]["time"]
                    words.append({
                        "word": curr_word,
                        "time": curr_chars[0]["time"],
                        "duration": curr_chars[-1]["time"] + curr_chars[-1]["duration"] - curr_chars[0]["time"],
                        "chars": curr_chars,
                    })
                curr_word = ""
                curr_chars = []
            else:
                curr_word += token
                curr_chars.append({
                    "char": token,
                    "time": tensor2time(timesteps[i]),
                    "duration": tensor2time(timesteps[i+1] - timesteps[i]) if i < len(timesteps) - 1 else 0,
                })
        return words


    def merge_paragraph_transcripts(self, transcripts):
        start_time = 0
        for transcript in transcripts:
            transcript['start'] = start_time
            for word in transcript['words']:
                word['time'] += start_time
                for c in word.get('chars', []):
                    c['time'] += start_time
            start_time += transcript['duration']
        return transcripts
    

    def merge_sentence_transcripts(self, transcripts):
        start_time = 0
        words = []
        sentence = ""
        duration = 0
        edit_distance = 0
        wer = 0
        errors = 0
        fatal_errors = 0
        for transcript in transcripts:
            transcript['start'] = 0
            duration += transcript['duration']
            sentence += transcript['sentence'] + ' '
            edit_distance += transcript['edit_distance']
            # wer += transcript['WER']
            errors += transcript['errors']
            fatal_errors += transcript['fatal_errors']
            for word in transcript['words']:
                word['time'] += start_time
                words.append(word)
                for c in word.get('chars'):
                    c['time'] += start_time
            start_time += transcript['duration']
        edit_distance /= len(transcripts)
        wer /= len(transcripts)
        return {
            'sentence': sentence,
            'start': 0,
            'duration': duration,
            'end': duration,
            'words': words,
            'edit_distance': edit_distance,
            # 'WER': wer,
            'errors': errors,
            'fatal_errors': fatal_errors
        }
        