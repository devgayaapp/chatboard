import os
from pathlib import Path
from components.screen import ImageSize, Movie, Screen
from components.video.video_generator import VideoChannelScreen, VideoWriter
from components.video.video_generator_pymovie import PymovieVideoWriter
from util.types import Config
import json
from config import OUT_DIR, MUSIC_DIR, OUT_DIR_MUSIC
import shutil





def parse_screens_to_jsons(movie: Movie, voice_generator, config: Config, logger):
    raise DeprecationWarning('voice generator is no longer in use')
    dirname = os.path.join(OUT_DIR, config['video_name'])
    logger.info(dirname)
    try:
        os.mkdir(dirname)
    except FileExistsError as e:
        shutil.rmtree(dirname)
        os.mkdir(dirname)

    with open(os.path.join(dirname, f"{config['video_name']}_nerative.json"), 'w') as f:
        screen_list = []
        for i, screen in enumerate(movie):
            try: 
                video_json = screen_to_json(f"{config['video_name']}_screen_{i}", screen, voice_generator, config)
                screen_list.append(video_json)
            except Exception as e:
                print(f'Error in movie: {movie.title}')
                raise e
        
        total_duration = sum([s['duration'] for s in screen_list])
        movie_nerative = {
            'name': config['video_name'], 
            'screens': screen_list,
            'duration': total_duration,
            'music': add_music_track(total_duration, config)
        }
        j = json.dump(movie_nerative, f, indent= 4, sort_keys=True)
        # f.write(j)
        return movie_nerative
    
        

def screen_to_json(id: str, screen: Screen, voice_generator, config: Config):
    raise DeprecationWarning('voice generator is no longer in use')
    video_writer = PymovieVideoWriter(config['fps'], config['cap_size'])
    if screen.images:
        audio_channel = voice_generator.create_audio_chanel_from_text(screen.text.text, pause_before=0.5, pause_after=1)
        # audio_channel = voice_generator.create_audio_chanel_from_text(screen.text.text)
        assert(audio_channel.duration > 0)
        video_json = video_writer \
            .start() \
            .images(screen.images) \
            .animation('expand') \
            .duration(audio_channel.duration) \
            .subtitles(screen.text.text) \
            .download_media(OUT_DIR / config['video_name']) \
            .to_json()
        print(video_json)
        #fixing paths
        for img in video_json['images']:
            img['filename'] = str(Path(img['filename']).relative_to(OUT_DIR))

    elif screen.text:
        audio_channel = voice_generator.create_audio_chanel_from_text(screen.text.text, pause_before=1, pause_after=1)
        # audio_channel = voice_generator.create_audio_chanel_from_text(screen.text.text)
        video = video_writer \
            .start() \
            .background(color=(0,0,0)) \
            .duration(audio_channel.duration) \
            .subtitles(screen.text.text) 
            
        if screen.title:
            video.title(screen.title, font_scale=3, thickness=4)            
        video_json = video.to_json()

    elif screen.title:
        audio_channel = voice_generator.create_audio_chanel_from_text(screen.title.text, pause_before=1, pause_after=1)
        assert(audio_channel.duration > 0)
        video_json = video_writer \
        .start() \
        .title(screen.title.text, font_scale=3, thickness=4) \
        .background(color=(0,0,0)) \
        .duration(audio_channel.duration) \
        .to_json()

    else:
        raise Exception('bad screen')
    
    audio_filename = f'{id}.mp3'
    audio_channel.tensor_to_file(os.path.join(OUT_DIR, config['video_name'], audio_filename), to_mp3=True)
    video_json['audio_file'] = os.path.join(config['video_name'], audio_filename)
    return video_json



def add_music_track(duration, config):    
    filename = os.path.join('Funkymania.mpeg')
    filepath = os.path.join(MUSIC_DIR, filename)
    
    filepath_dst = os.path.join(OUT_DIR_MUSIC,'', filename)
    if not os.path.exists(filepath_dst):
        shutil.copy(filepath, filepath_dst)
    return os.path.join(filename)



def parse_screens_to_clips(movie: Movie, voice_generator, config):
    raise DeprecationWarning('voice generator is no longer in use')
    """
        legasy function for creating screens with py movie
    """
    clip_list = []
    for i, screen in enumerate(movie):
        clip = screen_to_clip(screen, voice_generator, config)
        clip_list.append(clip)
    return clip_list

def screen_to_clip(screen: Screen, voice_generator, config):
    raise DeprecationWarning('voice generator is no longer in use')
    video_writer = PymovieVideoWriter(config['fps'], config['cap_size'])
    if screen.images:
        audio_channel = voice_generator.create_audio_chanel_from_text(screen.text.text, pause_before=0.5, pause_after=1)
        # audio_channel = voice_generator.create_audio_chanel_from_text(screen.text.text)
        assert(audio_channel.duration > 0)
        video_channel = video_writer \
            .start() \
            .images(screen.images) \
            .animation('expand') \
            .duration(audio_channel.duration - 1.5) \
            .subtitles(screen.text.text) \
            .write()

    elif screen.text:
        audio_channel = voice_generator.create_audio_chanel_from_text(screen.text.text, pause_before=1, pause_after=1)
        # audio_channel = voice_generator.create_audio_chanel_from_text(screen.text.text)
        video = video_writer \
            .start() \
            .background(color=(0,0,0)) \
            .subtitles(screen.text.text) \
            .duration(audio_channel.duration - 2)
        if screen.title:
            video.title(screen.title, font_scale=3, thickness=4)            
        video_channel = video.write()

    elif screen.title:
        audio_channel = voice_generator.create_audio_chanel_from_text(screen.title.text, pause_before=1, pause_after=1)
        assert(audio_channel.duration > 0)
        video_channel = video_writer \
        .start() \
        .title(screen.title.text, font_scale=3, thickness=4) \
        .background(color=(0,0,0)) \
        .duration(audio_channel.duration - 3) \
        .write()

    else:
        raise Exception('bad screen')

    audioclip = audio_channel.get_audio_clip()
    video_channel.audio = audioclip
    return video_channel




def aggregate_movie(filename: str, movie: Movie, video_writer, voice_generator, media_aggregator):
    raise Exception('movie py is no longer in use')
    # audio_list, video_list = sceenes_to_inputs(movie, voice_generator, video_writer)
    # media_aggregator(audio_list, video_list, filename)
    # return movie


def save_to_video_file(filename: str, video_clip):
    
    video_clip.write_videofile(
        filename, 
        temp_audiofile='temp-audio.m4a', 
        remove_temp=True, 
        fps=30,
        codec='libx264',  
        audio_codec="aac")


def add_music_track_pymovie(filepath, video_clip, volume=0.2):
    raise Exception('movie py is no longer in use')

    # music_track = AudioFileClip(filepath)
    # music_track = music_track.subclip(0, video_clip.duration)
    # music_track = music_track.fx(afx.volumex, volume)

    # final_audio_track = CompositeAudioClip([video_clip.audio, music_track])

    # video_clip.audio = final_audio_track
    # return video_clip


def create_movie(filepath, idx=0, fps: int = 30 , cap_size: ImageSize = (1777, 1000)):
    raise Exception('movie py is no longer in use')

    # wordpress_file = WordPressFile(filepath)
    # article = wordpress_file[idx]
    # screens = nerative_builder(article)
    # movie = Movie(screens, title=article.title)

    # video_writer = PymovieVideoWriter(fps, cap_size)
    # voice_generator = VoiceGenerator()
    # voice_generator.init()
    
    # clip_list = parse_screens_to_clips(movie, voice_generator, video_writer)
    
    # video_clip = concatenate_videoclips(clip_list)
    # filename = os.path.join('out', f'{article.title}.mp4')
    # save_to_video_file(filename, video_clip)
    # # video_clip.write_videofile(
    # #     filename, 
    # #     temp_audiofile='temp-audio.m4a', 
    # #     remove_temp=True, 
    # #     fps=30,
    # #     codec='libx264',  
    # #     audio_codec="aac")
    # return video_clip, movie
    # print('finished')