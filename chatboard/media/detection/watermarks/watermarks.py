import numpy as np
import cv2
from components.image.detection_client import DetectionClient
from components.image.image import Image
from glob import glob

from components.media.media import Media
from components.video.video import Video




def gen_mask(page, mask_value=255, mask_color=0):
    page_height, page_width = page['dimensions']
    # mask = np.ones((page_height, page_width), dtype=np.uint8) * mask_color
    mask = np.full((page_height, page_width), mask_color, dtype=np.uint8)
    for block in page['blocks']:
        for line in block['lines']:
            for word in line['words']:
                point1, point2 = word['geometry']
                x1 = int(point1[0] * page_width)
                x2 = int(point2[0] * page_width)
                y1 = int(point1[1] * page_height)
                y2 = int(point2[1] * page_height)
                mask[y1:y2, x1:x2] = mask_value
    return mask




async def generate_watermark_mask(image: Image, detection: DetectionClient):
    ocr_result = await detection.ocr_image(image)
    mask = gen_mask(ocr_result['pages'][0])
    return mask

async def remove_image_watermark(image: Image, detection: DetectionClient):
    # ocr_result = await detection.ocr_image(image)    
    # mask = gen_mask(ocr_result['pages'][0])
    mask = await generate_watermark_mask(image, detection)
    image_np = image.np_arr

    dilatekernel = np.ones((5, 5), 'uint8')
    mask = cv2.dilate(mask, dilatekernel)
    output = cv2.inpaint(image_np, mask, 3, flags=cv2.INPAINT_NS)
    return Image.from_numpy(output)





import subprocess
import os
import tempfile
import random
import re

# Prepare output name
def get_output_name(input_file, output_file=None):
    file_no_ext, extension = os.path.splitext(input_file)
    def_name = f"{file_no_ext}_cleaned{extension}"
    return output_file if output_file else def_name

# Get first few key frames
def get_key_frames(input_file, max_frames=50):
    command = f'ffprobe -hide_banner -loglevel warning -select_streams v -skip_frame nokey -show_frames -show_entries frame=pkt_dts_time {input_file} | grep "pkt_dts_time="'
    result = subprocess.check_output(command, shell=True).decode().splitlines()
    return random.sample(result, min(max_frames, len(result)))

# Save them as images, in a temporary directory
def extract_frames(input_file, keyframes_time, timeout=60):
    tmpdir = tempfile.mkdtemp()
    counter = 0
    for i in keyframes_time:
        time = i.split("=")[1]
        if not re.match("^[0-9]+([.][0-9]+)?$", time):
            print(f"Skipping unrecognize timing: {i}")
            continue
        command = f'ffmpeg -y -hide_banner -loglevel error -ss {time} -i {input_file} -vframes 1 {tmpdir}/output_{counter}.png'
        subprocess.call(command, shell=True, timeout=60)
        counter += 1
    if counter < 2:
        print(f"{counter} frames extracted, need at least 2, aborting.")
        exit(1)
    return tmpdir

# Extracting watermark
async def extract_watermark(detection, tmpdir):
    filepaths = list(glob(f'{tmpdir}/*.png'))
    f = random.choice(filepaths)
    rep_img = Image.from_file(f)
    mask = await generate_watermark_mask(rep_img, detection)
    Image.from_numpy(mask).to_file(f"{tmpdir}/mask.png")
    # command = f'./get_watermark.py {tmpdir}'
    # subprocess.call(command, shell=True)

# Removing watermark in video
def remove_watermark(input_file, output_file, tmpdir, timeout=60):
    command = f'ffmpeg -hide_banner -loglevel warning -y -stats -i {input_file} -acodec copy -vf "removelogo={tmpdir}/mask.png" {output_file}'
    subprocess.call(command, shell=True, timeout=timeout)

# Clean up
def clean_up(tmpdir):
    command = f'rm -rf {tmpdir}'
    subprocess.call(command, shell=True)

# Main function
async def remove_video_watermark(detection, input_file, output_file=None, max_frames=50, timeout=60):
    output_file = get_output_name(input_file, output_file)
    keyframes_time = get_key_frames(input_file, max_frames)
    tmpdir = extract_frames(input_file, keyframes_time, timeout=timeout)
    await extract_watermark(detection, tmpdir)
    remove_watermark(input_file, output_file, tmpdir, timeout=timeout)
    clean_up(tmpdir)
    



async def remove_media_watermarks(media: Media, detection: DetectionClient):
    media = media.deep_copy()
    if media.type == "IMAGE":
        image = await remove_image_watermark(media._media, detection)
        media._media = image
        media.thumbnail = image.get_thumbnail()
        return media
    elif media.type == "VIDEO":        
        new_filepath = media._media.filepath.parent/ ("w_"+ media._media.filepath.name)
        await remove_video_watermark(detection, str(media._media.filepath), str(new_filepath))        
        video = Video.from_file(new_filepath)        
        video.shots = media._media.shots
        media._media = video
        media.thumbnail = video.get_thumbnail()
        return media