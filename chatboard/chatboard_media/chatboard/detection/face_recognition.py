from config import DATA_DIR, FACE_DETECTION_MODEL
import numpy as np
from components.detection.cv_utils import get_cap_params
from config import BASE_DIR
import copy

try:
    import cv2
    import pandas as pd
except ModuleNotFoundError:
    pass
except ImportError:
    pass

COORD_CLMS = ['box_xmin', 'box_ymin', 'box_width', 'box_height', 'left_eye_x',
        'left_eye_y', 'right_eye_x', 'right_eye_y', 'nose_x', 'nose_y','mouth_x', 'mouth_y', 'left_ear_x' ,'left_ear_y' ,'right_ear_x' ,'right_ear_y'
]   



def video_color_hists(image_list, hist_size=32, hist_range=(0, 256)):
    color_hists = []
    for im in image_list:
        chans = cv2.split(im)
        color_hists.append(cv2.calcHist([chans[1], chans[0]], [0, 1], None, [hist_size, hist_size],
                        [hist_range[0], hist_range[1], hist_range[0], hist_range[1]]))
        # color_hists.append(cv2.calcHist([im], [0, 1, 2], None, [32, 32],
        #                 [0, 256, 0, 256]))
    color_hists = np.array(color_hists)
    return color_hists


def get_image_changes_arr(image_list):    
    color_hists = video_color_hists(image_list)
    image_diffs_arr = np.diff(color_hists, axis=0)
    peaks = remove_fades_dissolves(np.linalg.norm(image_diffs_arr, axis=(1,2)))
    color_hists_diff_df = pd.DataFrame({
        'prev_frame': range(0, peaks.shape[0]),
        'next_frame': range(1, peaks.shape[0] + 1),
        'peaks': peaks, 
        # 'prev_frame': range(0, len(image_list)- 1),
        # 'next_frame': range(1, len(image_list)),
        # 'peaks': remove_fades_dissolves(np.linalg.norm(image_diffs_arr, axis=(1,2))), 
        # 'mean': np.mean(image_diffs_arr, axis=(1,2)),
        # 'std': np.std(image_diffs_arr, axis=(1,2)),
        # 'norm': np.linalg.norm(image_diffs_arr, axis=(1,2)),
    })
    return color_hists_diff_df, color_hists


def get_hist_entropy(color_hists):
    entropy_list = []
    for i in range(color_hists.shape[0]):
        total_pixels = np.sum(color_hists[i])
        probabilities = color_hists[i] / total_pixels
        # Filter out zero probabilities to avoid log(0)
        probabilities = probabilities[probabilities > 0]
        # Calculate the entropy
        entropy = -np.sum(probabilities * np.log2(probabilities))
        entropy_list.append(entropy)
    return entropy_list


def remove_fades_dissolves(peaks_array, beta=0.2, window_size = 15):
    # https://www-nlpir.nist.gov/projects/tvpubs/tvpapers03/ramonlull.paper.pdf
    peaks_conv_array = np.convolve(peaks_array, np.ones(window_size) * beta)
    sub_array = peaks_array - peaks_conv_array[0: peaks_array.shape[0]]
    sub_array[sub_array<0] = 0
    return sub_array[0:-1]



def get_event_ranges(changes_df, frame_count):
    event_ranges = [(0, int(changes_df.iloc[0]['prev_frame']))]
    for i in range(1, len(changes_df)):
        event_ranges.append((int(changes_df.iloc[i-1]['next_frame']), int(changes_df.iloc[i]['prev_frame'])))
    event_ranges.append((int(changes_df.iloc[-1]['next_frame']), int(frame_count - 1)))
    return event_ranges



async def get_face_events(face_recognizer, image_list, event_ranges, fps):
    # app = FaceAnalysis(name=FACE_DETECTION_MODEL, root=str(BASE_DIR / 'models/insightface'), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    # app.prepare(ctx_id=0, det_size=(640,640))
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
            faces = await face_recognizer.recognize_faces(image)
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
                        'box': face['bbox'].tolist(),
                        'box_xmin': float(face['bbox'][0]),
                        'box_ymin': float(face['bbox'][1]),
                        'box_width': float(face['bbox'][2] - face['bbox'][0]),
                        'box_height': float(face['bbox'][3] - face['bbox'][1]),
                        'landmark': face['landmark_2d_106'].tolist(),
                        'pose': face['pose'].tolist(),
                    },
                    # 'embedding': face['embedding'],
                    'gender': int(face['gender']),                    
                    'age': face['age'],
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





def sec2minutes(seconds):
    return f'{int(seconds // 60)}:{int(seconds % 60)}'




def merge_events(event1, event2, fps):
    unique_faces = list({int(f['label']): f for f in (event1['face_coords'] + event2['face_coords'])}.values())
    return {
        'start_frame': event1['start_frame'],
        'end_frame': event2['end_frame'],
        'start': event1['start_frame'] / fps,
        'end': event2['end_frame'] / fps,
        'duration': (event2['end_frame'] - event1['start_frame']) / fps,
        'face_labels':np.unique(event1['face_labels'] + event2['face_labels']).tolist(),
        'face_count': len(np.unique(event1['face_labels'] + event2['face_labels'])),
        'face_coords': unique_faces,
        'index': event2['index']
    }

def get_diff_color_hist(event1, event2, color_hists):
    colors1 = np.mean(color_hists[event1['start_frame']: event1['end_frame']], axis=0)
    colors2 = np.mean(color_hists[event2['start_frame']: event2['end_frame']], axis=0)    
    return np.linalg.norm(colors1 - colors2)


def filter_events_by_time(events, fps, min_duration_sec=0.5):
    event_copy = copy.deepcopy(events)
    time_filtered_events = []
    for idx in range(0, len(event_copy)):
        event = event_copy[idx]
        next_event = event_copy[idx + 1] if idx + 1 < len(event_copy) else None    
        if not next_event:
            time_filtered_events.append(event)
        elif event['duration'] < min_duration_sec or set(event['face_labels']) == set(next_event['face_labels']):
                # print(get_diff_color_hist(event, next_event, color_hists))
            event_copy[idx + 1] = merge_events(event, next_event, fps)
        else:        
            time_filtered_events.append(event)
    return time_filtered_events


def filter_events_by_color_diff(events, color_hists, fps, max_diff=20000):
    color_merged_events = []
    events_copy = copy.deepcopy(events)
    for idx in range(0, len(events_copy)):
        event = events_copy[idx]
        next_event = events_copy[idx + 1] if idx + 1 < len(events_copy) else None    
        if not next_event:
            color_merged_events.append(event)
        elif get_diff_color_hist(event, next_event, color_hists) < max_diff:
                # print(get_diff_color_hist(event, next_event, color_hists))
            events_copy[idx + 1] = merge_events(event, next_event, fps)
        else:        
            color_merged_events.append(event)
    return color_merged_events



def read_video_frames(filepath):
    cap = cv2.VideoCapture(str(filepath))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fps, frame_count, cap_width, cap_height = get_cap_params(cap)
    image_list = []
    success=True
    while success:
        success, image = cap.read()
        if not success:
            break
        image_list.append(image)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return image_list, fps, frame_count, cap_width, cap_height, cap
    




def generate_video_events(cap,  start_at, end_at):
    # image_list, fps, frame_count, cap_width, cap_height = read_video_frames(filepath)
    
    fps, _, _, _ = get_cap_params(cap)
    start_frame = int(start_at * fps)
    end_frame = int(end_at * fps)
    frame_count = end_frame - start_frame
    
    image_list = []
    success=True
    frame = start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    while success and frame < end_frame:
        success, image = cap.read()
        if not success:
            break
        image_list.append(image)
        frame += 1
    color_hists_df, color_hists = get_image_changes_arr(image_list)
    changes_df = get_key_changes(color_hists_df, fps, 5)
    event_ranges = get_event_ranges(changes_df, frame_count)
    events, face_storage = get_face_events(image_list, event_ranges, fps)
    ft_events = filter_events_by_time(events, fps)
    fc_events = filter_events_by_color_diff(ft_events, color_hists, fps)
    return fc_events




#========================== old statistical events ==========================




def get_key_changes(color_hists_df, fps, seconds_per_event=30, field='peaks'):
    # frame_num = len(color_hists_df) + 1
    # max_event_number = frame_num / ( seconds_per_event * fps)
    # quantile = 1 - max_event_number / frame_num    
    # treshold = color_hists_df[field].quantile(quantile)    
    treshold = color_hists_df.peaks.mean() + color_hists_df.peaks.std() * 2
    color_hists_df['time'] = color_hists_df['next_frame'] / fps
    top_changs_df = color_hists_df[color_hists_df[field] > treshold]
    
    return top_changs_df



def get_statistical_events(color_hists_df, df, fps):
    
    frame_num = len(color_hists_df) + 1
    # # treshold = color_hists_df['std'].mean() + color_hists_df['std'].std() * 10
    # seconds_per_event = 30
    # max_event_number = frame_num / ( seconds_per_event * fps)
    # quantile = 1 - max_event_number / frame_num
    # # print(quantile)
    # treshold = color_hists_df['std'].quantile(quantile)
    # # print('treshold', treshold)
    # top_changs_df = color_hists_df[color_hists_df['std'] > treshold]
    top_changs_df = get_key_changes(color_hists_df, fps)
    event_list = []

    frame_df = group_frames(df)

    for i in range(len(top_changs_df) + 1):
        start_frame = top_changs_df.iloc[i - 1]['next_frame'] if i != 0 else 0        
        
        end_frame = top_changs_df.iloc[i]['prev_frame'] if i != len(top_changs_df) - 1 else frame_num

        faces = []
        face_labels = frame_df[frame_df['frame'] == start_frame]
        if len(face_labels) > 0:
            for l in face_labels.iloc[0]['faces']:
                try:
                    coords = df[(df['frame'] == start_frame) & (df['label'] == l) ].iloc[0][COORD_CLMS]
                    faces.append({
                        'label': l,
                        'face': coords.to_dict()
                    })
                except Exception as e:
                    print(e)
                    print('idx: ', i, 'start frame: ', start_frame, 'label: ', l)        
        event_list.append({
            'start_frame': int(start_frame),
            'end_frame': int(end_frame),
            'start': start_frame / fps,
            'end': (end_frame) / fps,
            'duration': (end_frame - start_frame) / fps,
            # 'end': (end_frame + 1) / fps,
            # 'duration': (end_frame - start_frame + 1) / fps,
            'face_labels': list(face_labels),
            'face_count': len(faces),
            'face_coords': faces,
            'index': i
        })
    return event_list
        
        
    



def group_frames(df, frame_count):
    # frame_df = df.groupby('frame').count().reset_index()
    # frame_df['faces'] = df.groupby('frame')['label'].apply(list)
    # frame_df['face_num'] = frame_df['faces'].str.len()
    # frame_df['face_no_list'] = df.groupby('frame')['face_no'].apply(list)
    # # frame_df['face_num'] = frame_df['faces'].apply(lambda x: len(x))
    # frame_df['frame_group'] = (frame_df.label != frame_df.label.shift(2)).cumsum()
    # frame_df['frame_count'] = 1
    # return frame_df
    #---------------------------------------
    # frames = []
    # currnt_frame = 0
    # currnt_faces = {}
    # currnt_face_no = {}
    # currnt_group = 0
    # for i, row in df.iterrows():
    #     if currnt_frame != row['frame']:
    #         frames.append({
    #             'frame': currnt_frame,
    #             'faces': list(currnt_faces.keys()),
    #             'face_no_list': list(currnt_face_no.keys()),
    #             'face_num': len(currnt_faces.keys()),
    #             'frame_group': currnt_group
    #         })
    #         currnt_frame = int(row['frame'])
    #         if not row['label'] in currnt_faces:
    #             currnt_group += 1
    #             currnt_faces = {}
    #             currnt_face_no = {}
    #             #diffrent context
    #     currnt_faces[int(row['label'])] = True
    #     currnt_face_no[int(row['face_no'])] = True

    # frames_df = pd.DataFrame.from_dict(frames)
    # return frames_df
    #---------------------------------------
    COLUMNS = [
        'frame',
        'face_no',
        'label',
    ]
    frame_df = df.groupby('frame',as_index=False).count()[COLUMNS]
    labels_sq = df.groupby('frame')['label'].apply(list)
    # face_num_sq = frame_df['faces'].str.len()  
    face_no_list_sq = df.groupby('frame')['face_no'].apply(list)
    # frame_df['frame_group'] = 1    

    frames = []


    i = 0
    row = frame_df.iloc[i]    
    # for i, row in frame_df.iterrows():
    for f in range(frame_count + 1):
            
        if i < len(frame_df) and row['frame'] == f:
            frames.append({
                'frame': row['frame'],
                'faces': labels_sq.iloc[i],
                'face_no_list': face_no_list_sq.iloc[i],
                'face_num': len(labels_sq.iloc[i]),
                'frame_group': 0
            })
            i = i + 1
            row = frame_df.iloc[i] if i < len(frame_df) else None            
        else:
            frames.append({
                'frame': f,
                'faces': [],
                'face_no_list': [],
                'face_num': 0,
                'frame_group': 0
            })

    updated_frames_df = pd.DataFrame.from_dict(frames)    
    
    currnt_group = -1
    prev_faces = set([-1]) #the first set can be empty
    for i, row in updated_frames_df.iterrows():            
        if prev_faces != set(row['faces']): #event changed
            currnt_group += 1
        prev_faces = set(row['faces'])        
        updated_frames_df.loc[i, 'frame_group'] = currnt_group
            
    

    # frames_df = pd.DataFrame.from_dict(grouped_frames)
    return updated_frames_df
        
        



def gen_frame_durations(frame_df):
    frame_df['frame_count'] = 1
    event_df = frame_df.groupby('frame_group').agg({'frame': ['min', 'max'], 'frame_count': ['sum'], 'faces': ['first'],'face_num': ['first']}).reset_index()
    return event_df


def smooth_events(cap, event_df):
    fps, frame_count, cap_width, cap_height = get_cap_params(cap)
    smoothed_events = []
    def unpack_row(row):
        return {
            'frame_start': row['frame']['min'],
            'frame_end': row['frame']['max'],
            'faces': set(row['faces']['first']) if not row['faces']['first'] is None else set([]),
            'frame_count': row['frame_count']['sum'],
        }
    current_event = None
    event_start = None
    current_group = 0

    for i, row in event_df.iterrows():
        if not current_event:
            current_event = unpack_row(row)
            event_start = current_event['frame_start']
        else:
            next_event = unpack_row(row)
            #connect next event to current event
            if next_event['faces'] == current_event['faces'] or next_event['frame_end'] - next_event['frame_start'] < 25:
                current_event['frame_end'] = next_event['frame_end']
                current_event['frame_count'] = current_event['frame_end'] - current_event['frame_start']
            else:
                #close the current event and start a new one
                current_event['frame_start'] = event_start
                smoothed_events.append(current_event)
                current_event = next_event
                event_start = current_event['frame_start']
            if i == len(event_df) - 1:
                current_event['frame_start'] = event_start
                smoothed_events.append(current_event)
                
    #validate events are consecutive
    prev_start = 0
    prev_end = -1
    for i, event in enumerate(smoothed_events):
        if event['frame_start'] != prev_end + 1:
            raise Exception(f'Events are not consecutive. event: {i} previus ends: {prev_end} event {i} starts: {event["frame_start"]}')
            print('event {} not consecutive'.format(i))
        prev_end = event['frame_end']

    return pd.DataFrame.from_dict(smoothed_events)



def transform_events_to_seconds(cap, df, smoothed_events_df):
    fps, _, _, _ = get_cap_params(cap)
     
    event_list = [] 
    for idx, moment in smoothed_events_df[smoothed_events_df['frame_count'] > fps].iterrows():
        
        start_frame = int(moment['frame_start'])
        end_frame = int(moment['frame_end'])
        # group = int(moment['frame_group'])
        faces = []
        face_labels = moment['faces']
        face_count = len(face_labels)
        for l in face_labels:
            try:
                # coords = df[(df['frame'] == start_frame) & (df['label'] == l) ].iloc[0][COORD_CLMS]
                coords = df[(df['frame'] == start_frame) & (df['label'] == l) ][COORD_CLMS].mean()
                faces.append({
                    'label': l,
                    'face': coords.to_dict()
                })
            except Exception as e:
                print(e)
                print('idx: ', idx, 'start frame: ', start_frame, 'label: ', l)        
        event_list.append({
            # 'start_frame': start_frame,
            # 'end_frame': end_frame,
            # 'start': start_frame / fps,
            # 'end': (end_frame + 1) / fps,
            # 'duration': (end_frame - start_frame + 1) / fps,
            # # 'group': group,
            # 'face_labels': list(face_labels),
            # 'face_count': face_count,
            # 'face_coords': faces,
            # 'index': idx
            'start_frame': start_frame,
            'end_frame': end_frame,
            'start': start_frame / fps,
            'end': end_frame / fps,
            'duration': (end_frame - start_frame) / fps,
            # 'group': group,
            'face_labels': list(face_labels),
            'face_count': face_count,
            'face_coords': faces,
            'index': idx
        })
    return event_list




# def get_cap_events(cap):
#     fps, frame_count, cap_width, cap_height = get_cap_params(cap)
#     cap, face_list, image_list = detect_faces_video(cap=cap)
#     df = cluster_detections(cap, face_list)
#     frame_df = group_frames(df, int(frame_count))
#     event_df = gen_frame_durations(frame_df)
#     smoothed_events_df = smooth_events(cap, event_df)
#     events = transform_events_to_seconds(cap, df, smoothed_events_df)
#     return events



# def get_label_ranges(labels, to_df=False):
#     start_idx = 0
#     ranges = []
#     curr_label = labels[0]
#     for i in range(len(labels)):
#         # print(curr_label, ' ', labels[i])
#         if curr_label != labels[i] or i == len(labels) - 1:
#             ranges.append({
#                 'start': start_idx,
#                 'end': i-1,
#                 'label': labels[i]
#             })            
#             start_idx = i
#         curr_label = labels[i]
#     if to_df:
#         return pd.DataFrame.from_dict(ranges)
#     return ranges





