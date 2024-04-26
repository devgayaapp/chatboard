from dataclasses import dataclass
import numpy as np

import torch
import torchaudio


from components.text.util import sanitize_sentence

import pickle


from io import BytesIO


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


class ForceAligner:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
        self.model = bundle.get_model().to(self.device)
        self.sample_rate = bundle.sample_rate
        self.labels = bundle.get_labels()
        self.dictionary = {c: i for i, c in enumerate(self.labels)}


    def align(self, waveform, text, sample_rate, to_seconds=True, plot_alignment=False, plot_to_file=False):

        if type(waveform) == np.ndarray:
            waveform = torch.from_numpy(waveform.copy()).float()

        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)

        if waveform.ndim == 1:
            waveform = torch.stack([waveform, waveform], dim=0)

        text = sanitize_sentence(text)
        transcript = "|".join([w.upper() for w in text.split(" ")])

        with torch.inference_mode():
            emissions, _ = self.model(waveform.to(self.device))
            emissions = torch.log_softmax(emissions, dim=-1)
        
        emissions = emissions[0].cpu().detach()

        tokens = [self.dictionary[c] for c in transcript]
        trellis = get_trellis(emissions, tokens)
        path = backtrack(trellis, emissions, tokens)
        segments = merge_repeats(path, transcript)
        word_segments = merge_words(segments)


        # if plot_alignment:
        #     plt.ioff()
        #     fig = plot_alignments2(
        #         trellis,
        #         segments,
        #         word_segments,
        #         waveform[0],
        #         self.sample_rate,
        #         transcript,
        #         path,                
        #     )
        #     if plot_to_file:
        #         from io import BytesIO
        #         stats_image = BytesIO()
        #         plt.savefig(stats_image, format='png')
        #         stats_image.seek(0)
        #     else:
        #         plt.ion()
        #         plt.show()

        alignment_artifacts = {
            'trellis': trellis,
            'segments': segments,
            'word_segments': word_segments,
            'waveform': waveform,
            'sample_rate': self.sample_rate,
            'transcript': transcript,
            'path': path,                
        }
        
        alignment_artifacts_bytes = BytesIO()
        pickle.dump(alignment_artifacts, alignment_artifacts_bytes)
        alignment_artifacts_bytes.seek(0)
                
        
        if to_seconds:
            ratio = (waveform[0].shape[0] / self.sample_rate) / emissions.shape[0]
            # ratio = waveform.size(0) / (trellis.size(0) - 1)
            for s in word_segments:
                s.start = float(s.start * ratio)
                s.end = float(s.end * ratio)

        # if plot_alignment and plot_to_file:
        #     return word_segments, stats_image
        return word_segments, alignment_artifacts_bytes




def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis





def backtrack(trellis, emission, tokens, blank_id=0):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise ValueError(f"Failed to align at trellis: {j}")
    return path[::-1]



def merge_repeats(path, transcript):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            # print('merge_repeats', i1, i2)
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments



def merge_words(segments, separator="|"):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        # print('merge_words', i1, i2)
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words





# def plot_trellis_with_segments(trellis, segments, transcript, path):
#     # To plot trellis with path, we take advantage of 'nan' value
#     trellis_with_path = trellis.clone()
#     for i, seg in enumerate(segments):
#         if seg.label != "|":
#             trellis_with_path[seg.start + 1 : seg.end + 1, i + 1] = float("nan")

#     fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(16, 9.5))
#     ax1.set_title("Path, label and probability for each label")
#     ax1.imshow(trellis_with_path.T, origin="lower")
#     ax1.set_xticks([])

#     for i, seg in enumerate(segments):
#         if seg.label != "|":
#             ax1.annotate(seg.label, (seg.start + 0.7, i + 0.3), weight="bold")
#             ax1.annotate(f"{seg.score:.2f}", (seg.start - 0.3, i + 4.3))

#     ax2.set_title("Label probability with and without repetation")
#     xs, hs, ws = [], [], []
#     for seg in segments:
#         if seg.label != "|":
#             xs.append((seg.end + seg.start) / 2 + 0.4)
#             hs.append(seg.score)
#             ws.append(seg.end - seg.start)
#             ax2.annotate(seg.label, (seg.start + 0.8, -0.07), weight="bold")
#     ax2.bar(xs, hs, width=ws, color="gray", alpha=0.5, edgecolor="black")

#     xs, hs = [], []
#     for p in path:
#         label = transcript[p.token_index]
#         if label != "|":
#             xs.append(p.time_index + 1)
#             hs.append(p.score)

#     ax2.bar(xs, hs, width=0.5, alpha=0.5)
#     ax2.axhline(0, color="black")
#     ax2.set_xlim(ax1.get_xlim())
#     ax2.set_ylim(-0.1, 1.1)






# def plot_alignments(trellis, segments, word_segments, waveform, sample_rate):
#     trellis_with_path = trellis.clone()
#     for i, seg in enumerate(segments):
#         if seg.label != "|":
#             trellis_with_path[seg.start + 1 : seg.end + 1, i + 1] = float("nan")

#     fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(16, 9.5))

#     ax1.imshow(trellis_with_path[1:, 1:].T, origin="lower")
#     ax1.set_xticks([])
#     ax1.set_yticks([])

#     for word in word_segments:
#         ax1.axvline(word.start - 0.5)
#         ax1.axvline(word.end - 0.5)

#     for i, seg in enumerate(segments):
#         if seg.label != "|":
#             ax1.annotate(seg.label, (seg.start, i + 0.3))
#             ax1.annotate(f"{seg.score:.2f}", (seg.start, i + 4), fontsize=8)

#     # The original waveform
#     ratio = waveform.size(0) / (trellis.size(0) - 1)
#     ax2.plot(waveform)
#     for word in word_segments:
#         x0 = ratio * word.start
#         x1 = ratio * word.end
#         ax2.axvspan(x0, x1, alpha=0.1, color="red")
#         ax2.annotate(f"{word.score:.2f}", (x0, 0.8))

#     for seg in segments:
#         if seg.label != "|":
#             ax2.annotate(seg.label, (seg.start * ratio, 0.9))
#     xticks = ax2.get_xticks()
#     plt.xticks(xticks, xticks / sample_rate)
#     ax2.set_xlabel("time [second]")
#     ax2.set_yticks([])
#     ax2.set_ylim(-1.0, 1.0)
#     ax2.set_xlim(0, waveform.size(-1))









# def plot_alignments2(trellis, segments, word_segments, waveform, sample_rate, transcript, path):
#     trellis_with_path = trellis.clone()
#     for i, seg in enumerate(segments):
#         if seg.label != "|":
#             trellis_with_path[seg.start + 1 : seg.end + 1, i + 1] = float("nan")

#     fig, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=(16, 14.25))

#     ax1.imshow(trellis_with_path[1:, 1:].T, origin="lower")
#     ax1.set_xticks([])
#     ax1.set_yticks([])

#     for word in word_segments:
#         ax1.axvline(word.start - 0.5)
#         ax1.axvline(word.end - 0.5)

#     for i, seg in enumerate(segments):
#         if seg.label != "|":
#             ax1.annotate(seg.label, (seg.start, i + 0.3))
#             ax1.annotate(f"{seg.score:.2f}", (seg.start, i + 4), fontsize=8)

#     # The original waveform
#     ratio = waveform.size(0) / (trellis.size(0) - 1)
#     ax2.plot(waveform)
#     for word in word_segments:
#         x0 = ratio * word.start
#         x1 = ratio * word.end
#         ax2.axvspan(x0, x1, alpha=0.1, color="red")
#         ax2.annotate(f"{word.score:.2f}", (x0, 0.8))

#     for seg in segments:
#         if seg.label != "|":
#             ax2.annotate(seg.label, (seg.start * ratio, 0.9))
#     xticks = ax2.get_xticks()
#     plt.xticks(xticks, xticks / sample_rate)
    


#     ax2.set_xlabel("time [second]")
#     ax2.set_yticks([])
#     ax2.set_ylim(-1.0, 1.0)
#     ax2.set_xlim(0, waveform.size(-1))

#     ax3.set_title("Label probability with and without repetation")
#     xs, hs, ws = [], [], []
#     for seg in segments:
#         if seg.label != "|":
#             xs.append((seg.end + seg.start) / 2 + 0.4)
#             hs.append(seg.score)
#             ws.append(seg.end - seg.start)
#             ax3.annotate(seg.label, (seg.start + 0.8, -0.07), weight="bold")
#     ax3.bar(xs, hs, width=ws, color="gray", alpha=0.5, edgecolor="black")

#     xs, hs = [], []
#     for p in path:
#         label = transcript[p.token_index]
#         if label != "|":
#             xs.append(p.time_index + 1)
#             hs.append(p.score)

#     ax3.bar(xs, hs, width=0.5, alpha=0.5)
#     ax3.axhline(0, color="black")
#     ax3.set_xlim(ax1.get_xlim())
#     ax3.set_ylim(-0.1, 1.1)


#     return fig




