

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
except:
    pass

try:
    import cv2
    from cv2 import CAP_PROP_FPS, CAP_PROP_FRAME_COUNT, CAP_PROP_POS_FRAMES
    import pandas as pd
    from PIL import Image, ImageDraw
except:
    pass


import numpy as np



def image_grid(imgs, rows, cols, is_np=True):
    assert len(imgs) == rows*cols

    if is_np:
        imgs = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in imgs]

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def get_cap_params(cap):
    fps = cap.get(CAP_PROP_FPS)
    frame_count = cap.get(CAP_PROP_FRAME_COUNT)
    cap_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    return fps, frame_count, cap_width, cap_height

    

def cv2_imshow(a, **kwargs):
    a = a.clip(0, 255).astype('uint8')
    # cv2 stores colors as BGR; convert to RGB
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)

    return plt.imshow(a, **kwargs)






def cv2_imshow_draw(a, boxs=None):
    a = a.clip(0, 255).astype('uint8')
    # cv2 stores colors as BGR; convert to RGB
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots()
    ax.imshow(a)
    if boxs:
        for box in boxs:
            rect = patches.Rectangle((box['xmin'], box['ymin']), box['width'], box['height'], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    return plt.show()


def frame2Image(cap, frame = None, sec = None):
    if frame is None and not sec is None:
        fps = cap.get(CAP_PROP_FPS)
        frame = sec * fps
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    success, image = cap.read()
    if not success:
        raise Exception('could not read image')
    return image


def show_frame(cap, frame=None, sec=None, boxs=None):
    if frame is None and not sec is None:
        fps = cap.get(CAP_PROP_FPS)
        frame = sec * fps
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    success, image = cap.read()    
    cv2_imshow_draw(image, boxs)


def show_metadata(cap, frame=None, sec=None, faces=None):
    faces = faces if faces is not None else []
    if frame is None and not sec is None:
        fps = cap.get(CAP_PROP_FPS)
        frame = sec * fps
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    success, image = cap.read()
    boxs = [{'xmin': f['face']['box_xmin'], 'ymin': f['face']['box_ymin'], 'width': f['face']['box_width'], 'height': f['face']['box_height']} for f in faces]
    cv2_imshow_draw(image, boxs)





def trim_cv_video(outname: str, filepath: str, start_at: int, end_at:int):
    cap = cv2.VideoCapture(str(filepath))
    fps, frame_count, cap_width, cap_height = get_cap_params(cap)
    outpath = 'tmp/' + outname

    start_frame = int(start_at * fps)
    end_frame = int(end_at * fps)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(outpath, fourcc, int(fps), (int(cap_width), int(cap_height) ))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for i in range(end_frame - start_frame):
        ret, frame = cap.read()
        if ret==True:
            out.write(frame)
        else:
            break
    out.release()




def write_event_video(images, events, outpath: str, fps: int):
    events_df = pd.DataFrame(events)
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    height = images[0].shape[0]
    width = images[0].shape[1]

    event_font_pos = (int(width * 0.05), int(height * 0.9))
    
    size = (width, height)
    # size = (int(cap_width), int(cap_height))
    out = cv2.VideoWriter(outpath, fourcc, fps, size)

    # cv_video_writer.open(outpath, None, fps, (cap_width, cap_height))
    # out = cv2.VideoWriter(outpath, -1, 20.0, (640,480))
    # out = cv2.VideoWriter_fourcc(*'DIVX')
    # cap.set(cv2.CAP_PROP_POS_FRAMES, from_frame)
    for frame, orig_image in enumerate(images):
        image = orig_image.copy()
        event = events_df[(events_df['start_frame'] <= frame) & (events_df['end_frame'] >= frame)]
        if len(event) == 1:
            cv2.putText(
                    img=image, 
                    text=f"event {event.iloc[0]['index']}", 
                    org=event_font_pos, 
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, 
                    fontScale=3, 
                    color=(0, 255, 0),
                    thickness=4)
            for face in event.iloc[0]['face_coords']:
                x = int(face['face']['box_xmin'])
                y = int(face['face']['box_ymin'])
                w = int(face['face']['box_width'])
                h = int(face['face']['box_height'])
                cv2.rectangle(img=image, pt1=(x,y), pt2=(x+w, y+h), color=(0, 0, 255), thickness=4)
        elif len(event) == 0:
            cv2.putText(
                    img=image, 
                    text=f"no event found", 
                    org=event_font_pos, 
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, 
                    fontScale=1, 
                    color=(0, 255, 0),
                    thickness=2)
        elif len(event) > 1:
            raise Exception('more than one event')
        out.write(image)
    out.release()
    # cv2.destroyAllWindows()





def write_video(cap, from_frame, to_frame, outpath: str):
    
    fps, frame_count, cap_width, cap_height = get_cap_params(cap)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    is_initialized = True
    size = (int(cap_width), int(cap_height))
    out = cv2.VideoWriter(outpath, fourcc, fps, size)

    # cv_video_writer.open(outpath, None, fps, (cap_width, cap_height))
    # out = cv2.VideoWriter(outpath, -1, 20.0, (640,480))
    # out = cv2.VideoWriter_fourcc(*'DIVX')
    cap.set(cv2.CAP_PROP_POS_FRAMES, from_frame)
    for i in range(to_frame - from_frame):
    # while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            # frame = cv2.flip(frame,0)

            # write the flipped frame
            out.write(frame)

            # cv2.imshow('frame',frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
                # break
        else:
            break
    out.release()
    # cv2.destroyAllWindows()



MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return image