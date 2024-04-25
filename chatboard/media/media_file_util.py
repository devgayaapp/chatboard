
import os





video_extensions = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm']
image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif']

def gen_thumbnail_filename(filename):
    sanitized_filename, ext = os.path.splitext(filename)
    if ext.lower() in video_extensions:
        return f"thumbnail-{sanitized_filename}.png"
    return f"thumbnail-{sanitized_filename}{ext}"



def is_video(filename):
    _, ext = os.path.splitext(filename)
    return ext.lower() in video_extensions

def is_image(filename):
    _, ext = os.path.splitext(filename)
    return ext.lower() in image_extensions