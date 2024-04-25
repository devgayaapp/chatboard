# from threading import main_thread
# import pytest
# from components.screen import Screen, Movie
# from components.paragraph import Media, Paragraph, Title
# from components.video.video_generator import VideoWriter
# from components.voice.voice_generator import VoiceGenerator
# from components.voice.voice import AudioChannelScreen
# from components.wordpress import WordPressFile
# from components.media.media_aggregator import media_aggregator, parse_screens_to_clips
# from components.nerative.nerative_builider import nerative_builder
# # from components.video.video import Video
# from os import path
# import ffmpeg
# import numpy as np
# import torch
# import sys
# import os
# import glob
# import math
# from config import TEMP_DIR

# sys.path.append('.')

# CONTENT_NAME = 'test.xml'
# FILE_PATH = path.join('data', 'test_data', CONTENT_NAME)


# def test_xml_loads():
#     wordpress_file = WordPressFile(FILE_PATH)
#     assert(len(wordpress_file) == 3)

#     article = wordpress_file[0]
#     assert(article.title == 'Signs that your dog misses you')
#     assert('taking your dog to work is rarely an option' in article.content)
#     # print([type(e).__name__ for e in article.elements])
#     assert len([e for e in article.elements if type(e) == Media]) == 2
#     assert len([e for e in article.elements if type(e) == Title]) == 6
#     assert len([e for e in article.elements if type(e) == Paragraph]) == 8
#     # assert(len(article.images) == 2)




# def test_images():
#     wordpress_file = WordPressFile(FILE_PATH, lazy_load=True)
#     article = wordpress_file[0]
#     image: Media = article.elements[3] 
#     assert type(image) == Media 
#     assert(image.src != None)
#     assert(image._mat == None)
#     assert(image.mat is not None)
#     print(image.size)
#     assert(image.size == (633, 955))
#     resized_image = image.resize((200, 200))
#     assert(resized_image.size == (200, 200))


# def test_media_download():
#     video_url = 'https://drive.google.com/file/d/1-DQM2R39RUbZfpOUZ4BH4Xqk1QPEb04T/view?usp=sharing'
#     media = Media(video_url)
#     metadata = media.get_metadata()
#     assert metadata['mimeType'] == 'video/mp4'
#     media.populate_media()
#     assert media.filename is not None
#     assert media.filename.suffix == '.mp4'


# def test_media_image_download():
#     image_url = 'https://static.wikia.nocookie.net/marvel_dc/images/4/4b/Batman_Vol_3_86_Textless.jpg/revision/latest?cb=20200502132734'
#     img_media = Media(image_url)
#     img_metadata = img_media.get_metadata()
#     assert img_metadata['mimeType'] == 'image/jpeg'
#     img_media.populate_media()
#     assert img_media.filename is not None
#     assert img_media.filename.suffix == '.jpg'




# def test_voice_adding():
#     wordpress_file = WordPressFile(FILE_PATH)
#     article = wordpress_file[0]
#     voice_generator = VoiceGenerator()
#     voice_generator.init()
#     assert(voice_generator.text2speech != None)
#     voice_generator.parse_article(article)
#     assert(article.elements[0].voice != None)
#     assert(article.elements[0].voice.duration > 26)
#     assert(type(article.elements[0].voice.wav) == torch.Tensor)
#     assert(article.elements[0].voice.duration > 1)
#     # article.elements[0].voice.wav.save(f'{TEMP_DIR}/test.wav')
#     # article.elements[0].voice.tensor_to_file(f'{TEMP_DIR}/test.mp3')
#     # assert(path.exists(f'{TEMP_DIR}/test.mp3'))

    
    


# def test_generate_screens():
#     wordpress_file = WordPressFile(FILE_PATH)
#     article = wordpress_file[0]
#     voice_generator = VoiceGenerator()
#     voice_generator.init()
#     voice_generator.parse_article(article)

#     title = article.elements[1]
#     text = article.elements[2]
#     image = article.elements[3]
#     print(image.size)
#     screens = nerative_builder(article)
#     assert(len(screens) > 0)
#     assert(screens[0].text != None)
#     assert(screens[0].title == None)
#     assert(screens[1].text == None)
#     assert(screens[1].title != None)
#     print(screens)
#     # screen = Screen(text, [image], title)

# def test_big_file_sceen_generation():
#     files = []
#     for f in glob.glob(os.path.join('data', '*.xml')):
#         wordpress_file = WordPressFile(f)
#         files.append(wordpress_file)
#     assert(len(files) == 2)

    
    
# def test_generate_voice_channels():
#     wordpress_file = WordPressFile(FILE_PATH)
#     article = wordpress_file[0]
#     screens = nerative_builder(article)
#     movie = Movie(screens, article.title)
#     voice_generator = VoiceGenerator()
#     voice_generator.init()

#     screen1 = movie[0]
#     screen2 = movie[1]

#     assert(screen1.title is not None)
#     assert(screen2.text is not None)
    
#     voice_text = voice_generator.create_voice(screen2.text.text)
#     assert(voice_text.duration > 0)
#     # voice_list = [voice_text, voice_title]
#     text_audio_channel = AudioChannelScreen(voice_text)
#     assert(text_audio_channel.file.name is not None)
#     text_audio_channel.tensor_to_file(f'{TEMP_DIR}/text.mp3', to_mp3=True)
#     text_audio_channel.tensor_to_file(f'{TEMP_DIR}/text.wav')
#     # assert(text_audio_channel.input is not None)
#     padded_audio_chanel = voice_generator.create_audio_chanel_from_text(screen2.text.text, pause_before=3, pause_after=5)
#     assert(round(padded_audio_chanel.duration) == (round(text_audio_channel.duration) + 3 + 5))



# def test_generate_video_channels():
#     wordpress_file = WordPressFile(FILE_PATH)
#     article = wordpress_file[0]
#     screens = nerative_builder(article)
#     movie = Movie(screens, article.title)

#     screen_text = movie[0]
#     screen_title = movie[1]
#     screen_text_image = movie[2]


    

#     fps = 30
#     min_width = 300
#     min_height = int(( min_width / 9 ) * 16)
#     cap_size = (min_height, min_width)

#     video_writer = VideoWriter(fps, cap_size)
    
#     # title_video_channel = video_writer.write(title=screen_title.title.text, duration=10, background='black')
#     # images_video_channel = video_writer.write(images=screen_text_image.images, duration=10, animation='expand')

#     images_video_channel = video_writer \
#         .start() \
#         .images(screen_text_image.images) \
#         .animation('expand') \
#         .duration(15) \
#         .write()

#     res1 = ffmpeg.probe(images_video_channel.filename)
#     assert(int(float(res1['format']['duration'])) == 15)

#     title_video_channel = video_writer \
#         .start('test.avi') \
#         .title(screen_title.title.text) \
#         .background(color='black') \
#         .duration(15) \
#         .write()
   
#     res2 = ffmpeg.probe(title_video_channel.filename)
#     assert(int(float(res1['format']['duration'])) == 15)


# def test_video_text_writing():
#     text = 'adsf asdf sadf asdf  fasd safd  safd safd safd safd safd safd safd safd safd safd safd safd safd safd safd safd safd safd'
#     fps  = 30
#     screen_width = 900
#     screen_height = int(( screen_width / 9 ) * 16)
#     cap_size = (screen_height, screen_width)

#     video_writer = VideoWriter(fps, cap_size)

#     # thickness = 3
#     # font_scale = 2

#     img1 = video_writer._create_color_image(video_writer._cap_size, (0,0,0))
#     img2 = video_writer._create_color_image(video_writer._cap_size, (0,0,0))
#     video_writer._write_text(text, img1, thickness=3, font_scale=1)
#     video_writer._write_text(text, img2, thickness=3, font_scale=1)
#     assert np.array_equal(img1, img2)
#     img3 = video_writer._create_color_image(video_writer._cap_size, (0,0,0))
#     video_writer._write_text(text, img2, thickness=1, font_scale=2)
#     assert not np.array_equal(img1, img3)
            
    


#     # for screen in movie:
#     #     voice_list = []
#     #     if screen.title:
#     #         voice = voice_generator.create_voice(screen.title)
#     #         voice_list.append(voice)
#     #     if screen.text:
#     #         voice = voice_generator.create_voice(screen.text)
#     #         voice_list.append(voice)
#     #     audio_channel = AudioChannelScreen(voice_list)

        
        
    

    
    
