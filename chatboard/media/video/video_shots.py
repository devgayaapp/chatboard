from pathlib import Path

from components.detection.face_recognition import get_cap_params, get_hist_entropy, get_key_changes, get_image_changes_arr, get_event_ranges, get_face_events, filter_events_by_time, filter_events_by_color_diff


from config import DEBUG
from time import time
import numpy as np
import components.image.image as comp_image
import copy

try:
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    import pandas as pd
except ModuleNotFoundError:
    print("video events not available")
    pass

print(DEBUG)
if DEBUG == True or DEBUG == 'True':
    try:
        import IPython.display as ipd
        from components.detection.cv_utils import image_grid, cv2_imshow
    except:
        pass

def scale_face(person, scale):
    person_copy=copy.copy(person)
    face = person_copy['face']
    scaled_face = {}
    scaled_face['box'] = [c * scale for c in face['box']]
    scaled_face['box_xmin'] = face['box_xmin'] * scale
    scaled_face['box_ymin'] = face['box_ymin'] * scale
    scaled_face['box_width'] = face['box_width'] * scale
    scaled_face['box_height'] = face['box_height'] * scale
    scaled_face['landmark'] = [[l[0]*scale, l[1]*scale] for l in face['landmark']] 
    scaled_face['pose'] = [p*scale for p in face['pose']]
    person_copy['face'] = scaled_face
    return person_copy


class Shot:

    def __init__(self, event, event_df, frame_list, frame_entropy) -> None:
        self.start_frame = event['start_frame']
        self.end_frame = event['end_frame']
        self.start_time = event['start']
        self.end_time = event['end']
        self.duration = event['duration']
        self.face_labels = event['face_labels']
        self.face_count = event['face_count']
        self.face_coords = event['face_coords']
        self.index = event['index']
        self.event_df = event_df
        self.frame_list = frame_list
        self.frame_entropy = frame_entropy
        self._rep_image = None
        self.metadata = None
        self.embeddings = None
        self.vector_uuid = None
        # print('event created')


    def scale(self, scale):
        event_copy = copy.copy(self)
        event_copy.face_coords = [scale_face(p, scale) for p in event_copy.face_coords]
        # event_copy.frame_list = None
        return event_copy


    # def get_metadata(self):
    #     metadata = {}
    #     metadata.update(self.metadata)
    #     metadata['index'] = self.index
    #     metadata['start_time'] = self.start_time
    #     metadata['duration'] = self.duration
    #     metadata['video'] = True
    #     return metadata
        
    @property
    def width(self):
        return self.frame_list[0].shape[1]
    
    @property
    def height(self):
        return self.frame_list[0].shape[0]


    def get_image_grid(self):
        imgs = [self.frame_list[0], self.frame_list[-1]]
        return image_grid(imgs, 1, 2)


    def get_image(self):
        if self._rep_image is None:
            # img_idx = (self.end_frame - self.start_frame) // 2
            max_ent_idx = np.argmax(self.frame_entropy)
            pil_img = Image.fromarray(cv2.cvtColor(self.frame_list[max_ent_idx], cv2.COLOR_BGR2RGB))        
            self._rep_image = comp_image.Image.from_pil(pil_img)
        return self._rep_image


    def get_start_end_images(self):
        np_imgs = [(self.start_frame, self.start_time, self.frame_list[0]),(self.end_frame, self.end_time, self.frame_list[-1])]
        pil_imgs = []
        font = ImageFont.truetype("UbuntuMono-RI.ttf", 60)
        # font = ImageFont.FreeTypeFont(size=16)
        # font = ImageFont.load("arial.pil")

        for im_f, im_t, img in np_imgs:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            pil_img_draw = ImageDraw.Draw(pil_img)
            pil_img_draw.rectangle([(0,0), (250, 200)], fill='gray', width=2)
            pil_img_draw.text((0,0), f"Shot {self.index}", font=font, fill='white')                
            pil_img_draw.text((0,60), f"f {im_f}", font=font, fill='yellow')
            pil_img_draw.text((0,120), f"t {round(im_t,2)}", font=font, fill='cyan', )
            for face in self.face_coords:
                pil_img_draw.rectangle(face['face']['box'], outline='red', width=2)

                
            pil_imgs.append(pil_img)
        return pil_imgs
    
    def to_json(self):
        return {
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'start': self.start_time,
            'end': self.end_time,
            'duration': self.duration,
            'face_count': self.face_count,
            'face_labels': self.face_labels,
            'index': self.index,
            'face_coords': [{
                'label': f['label'],
                'face': {
                    'box': list(f['face']['box']),
                    'box_xmin': f['face']['box_xmin'],
                    'box_ymin': f['face']['box_ymin'],
                    'box_width': f['face']['box_width'],
                    'box_height': f['face']['box_height'],
                    'pose': list(f['face']['pose']),
                },
                'gender': f['gender'],
                'age': f['age'],
            } for f in self.face_coords]
        }
    
    @staticmethod
    def from_json(data):
        return Shot(data, None, None, None)
        


    def _repr_html_(self):
        # imgs = [self.frame_list[0], self.frame_list[-1]]
        if self.frame_list is not None:
            imgs = self.get_start_end_images()
            ipd.display(image_grid(imgs, 1, 2, is_np=False))
        return f'''
<div>
    <div>
        <span>Start time: {round(self.start_time,2)}</span>
        <span>End time: {round(self.end_time,2)}</span>
        <span> start frame: {self.start_frame}</span>
        <span> end frame: {self.end_frame}</span>
    </div>
    <div>
        <span>Duration: {self.duration}</span>
        <span>Face Count: {self.face_count}</span>
        <span>Face Labels: {self.face_labels}</span>
    </div>
</div>
        '''
        # cv2_imshow(self.frame_list[0])
        # return self.event_df.to_html()

    


class ShotList:

    def __init__(self, events) -> None:
        self.events = events

    
    def _repr_html_(self):
        imgs = []
        for event in self.events:
            imgs += event.get_start_end_images()
        return image_grid(imgs, len(self.events), 2, is_np=False)
        # ipd.display(image_grid(imgs, len(self.events), 2, is_np=False))

    def __getitem__(self, key):
        return self.events[key]
            # event._repr_html_()
        # return self.event_df.to_html()


class VideoShots:


    def __init__(self, video_filepath):
        self.is_initialized = False
        self.video_filepath = video_filepath
        cap = cv2.VideoCapture(self.video_filepath)
        fps, frame_count, cap_width, cap_height = get_cap_params(cap)
        self.fps = fps
        self.frame_count = frame_count
        self.cap_width = cap_width
        self.cap_height = cap_height
        self.duration = frame_count / fps
        self.frame = 0
        self.generate_image_frames(0, self.frame_count, cap)
        self.iter_idx = 0
        self.event_list = None
        # color_hists_df, color_hists = get_image_changes_arr(self.image_list)
        # self.color_hists_df = color_hists_df
        # self.color_hists = color_hists
        # self.changes_df = get_key_changes(color_hists_df, fps, 5)
        # self.event_ranges = get_event_ranges(self.changes_df, frame_count)
        # events, face_storage = get_face_events(self.image_list, self.event_ranges, fps)
        # self.events = events
        # self.face_storage = face_storage
        # ft_events = filter_events_by_time(self.events, fps)
        # fc_events = filter_events_by_color_diff(ft_events, color_hists, fps)
        # self.filtered_events = fc_events

    def get_vector_uuids(self):
        if self.event_list is None:
            return None
        return [e.vector_uuid for e in self.event_list]


    def set_fc_events(self, fc_events):
        self.filtered_events_df = pd.DataFrame(fc_events)
        self.is_initialized = True

    
    def generate_image_frames(self, start_frame, end_frame, cap):
        image_list = []
        success=True
        self.frame = start_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame)
        while success and self.frame < end_frame:
            success, image = cap.read()
            if not success:
                raise Exception("Could not read frame")
                # break
            image_list.append(image)
            self.frame += 1
        self.image_list = image_list


    def __getitem__(self, key)-> Shot:
        if not self.is_initialized or not self.event_list:
            raise Exception("Shots not initialized")
        return self.event_list[key]
        
    
    def init_events(self):
        events = []
        for index, row in self.filtered_events_df.iterrows():
            event = Shot(row, self.filtered_events_df, self.image_list[row['start_frame']:row['end_frame']], self.frame_entropy[row['start_frame']:row['end_frame']])
            events.append(event)
        self.event_list = events
        return self.event_list


    def __len__(self):
        return len(self.filtered_events_df)
    # def to_json(self):
    #     return self.filtered_events_df.to_dict(orient='records')

    def __iter__(self):
        if not self.event_list:
            self.init_events()
        self.iter_idx = 0
        return self
    
    def __next__(self):
        try:
            # item = self.__getitem__(self.iter_idx)
            # self.event_list[self.iter_idx]
            event = self.event_list[self.iter_idx]
            self.iter_idx += 1
            return event
        except IndexError:
            raise StopIteration


    def to_json(self):
        video_events = []
        for index, row in self.filtered_events_df.iterrows():
            video_events.append({
                'start_frame': row['start_frame'],
                'end_frame': row['end_frame'],
                'start': row['start'],
                'end': row['end'],
                'duration': row['duration'],
                'face_count': row['face_count'],
                'face_labels': list(row['face_labels']),
                'index': row['index'],
                'face_coords': [{
                    'label': f['label'],
                    'face': {
                        'box': list(f['face']['box']),
                        'box_xmin': f['face']['box_xmin'],
                        'box_ymin': f['face']['box_ymin'],
                        'box_width': f['face']['box_width'],
                        'box_height': f['face']['box_height'],
                        'pose': list(f['face']['pose']),
                    },
                    'gender': f['gender'],
                    'age': f['age'],
                } for f in row['face_coords']]
            })
        return video_events

    



async def get_face_events(detection, image_list, event_ranges, fps):        
    face_list=[]

    EVERY_NTH_FRAME = 30
    FACE_SIM_TRESH = 0.7
    face_storage = []
    events = []

    image_list_with_faces = []

    def get_face_id(face, face_storage):
        max_face_id = -1
        if len(face_storage) == 0:
            face_storage.append(face.normed_embedding)
            max_face_id=0
        else:
            similarity_values = [face.normed_embedding.dot(f) for f in face_storage]
            print(similarity_values)
            max_face_id = int(np.argmax(similarity_values))
            if similarity_values[max_face_id] < FACE_SIM_TRESH:
                face_storage.append(face.normed_embedding)
                max_face_id = len(face_storage) - 1
        return max_face_id


    for i, event_range in enumerate(event_ranges):
        # for frame in range(event[0], event[1], EVERY_NTH_FRAME):
            frame = event_range[0]
            image=image_list[frame]
            faces = await detection.arecognize_faces(image)
            image_faces = []
            for face in faces:
                # max_face_id = -1
                # if len(face_storage) == 0:
                #     face_storage.append(face.normed_embedding)
                #     max_face_id=0
                # else:
                #     similarity_values = [face.normed_embedding.dot(f) for f in face_storage]
                #     print(similarity_values)
                #     max_face_id = np.argmax(similarity_values)
                #     if similarity_values[max_face_id] < FACE_SIM_TRESH:
                #         face_storage.append(face.normed_embedding)
                #         max_face_id = len(face_storage) - 1            
                max_face_id = get_face_id(face, face_storage)
                image_faces.append({
                    'label': max_face_id,
                    'face': {
                        'box': face.bbox.tolist(),
                        'box_xmin': float(face.bbox[0]),
                        'box_ymin': float(face.bbox[1]),
                        'box_width': float(face.bbox[2] - face.bbox[0]),
                        'box_height': float(face.bbox[3] - face.bbox[1]),
                        'landmark': face.landmark_2d_106.tolist(),
                        'pose': face.pose.tolist(),
                    },
                    # 'embedding': face['embedding'],
                    'gender': int(face.gender),                    
                    'age': face.age,
                    })
            # face_list.append({
            #     'frame': frame,
            #     'faces': image_faces
            # })
            # image_list_with_faces.append(app.draw_on(image, faces))
            events.append({
                'start_frame': event_range[0],
                'end_frame': event_range[1],
                'start': event_range[0] / fps,
                'end': event_range[1] / fps,
                'duration': (event_range[1] - event_range[0]) / fps,
                'face_labels': [int(f['label']) for f in image_faces],
                'face_count': len(image_faces),
                'face_coords': image_faces,
                'index': i
            })

    return events, face_storage


async def get_video_shots(filepath, face_recognizer, logger=None, output_events=True):
    filepath = str(filepath)
    # yt = YoutubeVideo(url, video_name, start_at, end_at)
    # if logger:
    #     logger.info(f"Downloading video for {yt.media_name}")
    # yt.get_video()
    start_time = time()
    # if logger:
    #     logger.info(f'video downloaded {yt.video_filepath} took {download_time - start_time} seconds')

    media_name = Path(filepath).name
    #--------
    video_events = VideoShots(filepath)
    color_hists_df, color_hists = get_image_changes_arr(video_events.image_list)
    if logger:
        logger.info(f"Getting image changes for {media_name}")
    video_events.color_hists_df = color_hists_df
    video_events.color_hists = color_hists
    video_events.frame_entropy = get_hist_entropy(color_hists)
    if logger:
        logger.info(f"Getting key changes for {media_name}")
    video_events.changes_df = get_key_changes(color_hists_df, video_events.fps, 5)
    if logger:
        logger.info(f"Getting event ranges for {media_name}")
    video_events.event_ranges = get_event_ranges(video_events.changes_df, video_events.frame_count)
    if logger:
        logger.info(f"detecting faces for {media_name}")
    events, face_storage = await get_face_events(face_recognizer, video_events.image_list, video_events.event_ranges, video_events.fps)
    video_events.events = events
    video_events.face_storage = face_storage
    if logger:
        logger.info(f"processing events for {media_name}")
    ft_events = filter_events_by_time(video_events.events, video_events.fps)
    fc_events = filter_events_by_color_diff(ft_events, color_hists, video_events.fps)
    video_events.set_fc_events(fc_events)
    #--------
    event_time = time()
    if logger:
        logger.info(f'video events generated. took {event_time - start_time} seconds')
    file_path = Path(filepath)
    if logger:
        logger.info(f'uploading video {file_path.name}')

    output_json = {
        'video_key': file_path.name, 
        'events': video_events.to_json(), 
        'duration': video_events.duration, 
        'fps': video_events.fps, 
        'cap_width': video_events.cap_width, 
        'cap_height': video_events.cap_height
    }
    if output_events:
        return output_json, video_events
    return output_json