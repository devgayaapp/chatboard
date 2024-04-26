
import mediapipe as mp
# from sklearn.cluster import AgglomerativeClustering
import numpy as np
from config import BASE_DIR, DATA_DIR
from components.image.image import Image

try:
    import cv2
except ModuleNotFoundError:
    print("face detection won't work without cv2")
    pass
# from components.detection.cv_utils import get_cap_params

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

COORD_CLMS = ['box_xmin', 'box_ymin', 'box_width', 'box_height', 'left_eye_x',
        'left_eye_y', 'right_eye_x', 'right_eye_y', 'nose_x', 'nose_y','mouth_x', 'mouth_y', 'left_ear_x' ,'left_ear_y' ,'right_ear_x' ,'right_ear_y'
]   

def detect_faces_mesh(img, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    mp_face_mesh = mp.solutions.face_mesh
    faces = []
    with mp_face_mesh.FaceMesh(
        max_num_faces=5,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence) as face_mesh:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(imgRGB)

        

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                h, w, c = img.shape
                cx_min=  w
                cy_min = h
                cx_max= cy_max= 0
                for id, lm in enumerate(faceLms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if cx<cx_min:
                        cx_min=cx
                    if cy<cy_min:
                        cy_min=cy
                    if cx>cx_max:
                        cx_max=cx
                    if cy>cy_max:
                        cy_max=cy
                faces.append((cx_min, cy_min, cx_max-cx_min, cy_max-cy_min))
    return faces

def sanitize_coords(coords):
    return max(int(coords), 0)


def crop_image_face(image, box):
    x = sanitize_coords(box.xmin * image.shape[1])
    y = sanitize_coords(box.ymin * image.shape[0])
    w = sanitize_coords(box.width * image.shape[1])
    h = sanitize_coords(box.height * image.shape[0])
    # return image[x:x+w, y:y+h]
    return image[y:y+h, x:x+w]

hog = cv2.HOGDescriptor()

def extract_image_features(image, box, use_hog=False):
    try:
        face_img = crop_image_face(image, box)
        chans = cv2.split(face_img)
        color_hist = cv2.calcHist([chans[1], chans[0]], [0, 1], None, [32, 32],
                [0, 256, 0, 256])
    except Exception as e:
        print(e)
        # color_hist = np.zeros((32,32))
    fatures = {
        'color_hist_mean': np.mean(color_hist),
        'color_hist_std': np.std(color_hist),
        'color_hist_min': np.min(color_hist),
        'color_hist_max': np.max(color_hist),
    }
    if use_hog:
        hog_hist = hog.compute(face_img)
        fatures.update({
            'hog_hist_mean': np.mean(hog_hist),
            'hog_hist_std': np.std(hog_hist),
            'hog_hist_min': np.min(hog_hist),
            'hog_hist_max': np.max(hog_hist),
        })
    return fatures

def detect_faces_image(image, face_detection, draw=False, verbose=False, use_hog=False):
    """
        Parameters
        ---------
        image: cv2 image
        face_detection: mp_facedetection
        draw: bool add face drawings
    """
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)
    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_faces = []
    if results.detections: 
        if verbose:
            print('number of faces: ', len(results.detections))       
        for face_no, face in enumerate(results.detections):
            face_data = face.location_data
            img_face = {
                'box': face_data.relative_bounding_box,
                'left_eye': mp_face_detection.get_key_point(face, mp_face_detection.FaceKeyPoint.LEFT_EYE),
                'right_eye': mp_face_detection.get_key_point(face, mp_face_detection.FaceKeyPoint.RIGHT_EYE),
                'left_ear': mp_face_detection.get_key_point(face, mp_face_detection.FaceKeyPoint.LEFT_EAR_TRAGION),
                'right_ear': mp_face_detection.get_key_point(face, mp_face_detection.FaceKeyPoint.RIGHT_EAR_TRAGION),
                'nose': mp_face_detection.get_key_point(face, mp_face_detection.FaceKeyPoint.NOSE_TIP),
                'mouth': mp_face_detection.get_key_point(face, mp_face_detection.FaceKeyPoint.MOUTH_CENTER),
                'confidence': face.score,
                'face_no': face_no
            }
            hist_featurs= extract_image_features(image, face_data.relative_bounding_box, use_hog=use_hog)
            img_face.update(hist_featurs)
            image_faces.append(img_face)
            if draw:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)                    
    return image_faces, image


def detect_faces_image_sa(image, draw=False, verbose=False, confidence_tresh=0.5, use_hog=False):
    if type(image) == Image:
        image = image.np_arr[:, :, ::-1].copy() 
    with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=confidence_tresh) as face_detection:
        return detect_faces_image(image, face_detection, draw, verbose, use_hog=use_hog)


def detect_faces_video(filepath: str=None, cap=None, draw=False, get_images=False, use_hog=False):
    """
        Parameters
        -------
        file: str
            location of the file to process
        draw: bool
            add drawing to the video images and return image list. high memory.
    """
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    image_list = []
    face_list = []
    if cap is None and filepath is not None:
        cap = cv2.VideoCapture(str(filepath))
    if filepath is None and cap is None:
        raise ValueError('filepath or cap must be provided')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5) as face_detection:
        success, image = cap.read()
        frame_num = 0
        while success:
        
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image_faces, image_proccessed = detect_faces_image(image, face_detection, draw, use_hog=use_hog)
            face_list.append({
                'frame': frame_num,
                'faces': image_faces
            })
            if draw:
                image_list.append(image_proccessed)
            elif get_images:
                image_list.append(image)
            success, image = cap.read()
            frame_num+=1
            # Flip the image horizontally for a selfie-view display.
            # cv2_imshow(cv2.flip(image, 1))
            # cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
            # if cv2.waitKey(5) & 0xFF == 27:
            #   break
    return cap, face_list, image_list


# def get_cap_params(cap):
#     fps = cap.get(CAP_PROP_FPS)
#     frame_count = cap.get(CAP_PROP_FRAME_COUNT)
#     cap_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
#     cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
#     return fps, frame_count, cap_width, cap_height

    



# def cluster_detections(cap, face_list, use_hog=False):
#     fps, frame_count, cap_width, cap_height = get_cap_params(cap)
    
#     unpacked_faces = [{'face': f, 'frame': rec['frame']} for rec in face_list for f in rec['faces']]

#     def uppack_scaled_face(face):
#         return {
#             'box_xmin': sanitize_coords(face['box'].xmin * cap_width),
#             'box_ymin': sanitize_coords(face['box'].ymin * cap_height),
#             'box_width': sanitize_coords(face['box'].width * cap_width),
#             'box_height': sanitize_coords(face['box'].height * cap_height),
#             'left_eye_x': sanitize_coords(face['left_eye'].x * cap_width),
#             'left_eye_y': sanitize_coords(face['left_eye'].y * cap_height),
#             'right_eye_x': sanitize_coords(face['right_eye'].x * cap_width),
#             'right_eye_y': sanitize_coords(face['right_eye'].y * cap_height),
#             'left_ear_x': sanitize_coords(face['left_ear'].x * cap_width),
#             'left_ear_y': sanitize_coords(face['left_ear'].y * cap_height),
#             'right_ear_x': sanitize_coords(face['right_ear'].x * cap_width),
#             'right_ear_y': sanitize_coords(face['right_ear'].y * cap_height),
#             'nose_x': sanitize_coords(face['nose'].x * cap_width),
#             'nose_y': sanitize_coords(face['nose'].y * cap_height),
#             'mouth_x': sanitize_coords(face['mouth'].x * cap_width),
#             'mouth_y': sanitize_coords(face['mouth'].y * cap_height),
#             'face_no': face['face_no']
#         }
#     def uppack_face(face):
#         features = {
#             'box_xmin': face['box'].xmin,
#             'box_ymin': face['box'].ymin,
#             'box_width': face['box'].width,
#             'box_height': face['box'].height,
#             # 'left_eye_x': face['left_eye'].x,
#             # 'left_eye_y': face['left_eye'].y,
#             # 'right_eye_x': face['right_eye'].x,
#             # 'right_eye_y': face['right_eye'].y,
#             # 'left_ear_x': face['left_ear'].x,
#             # 'left_ear_y': face['left_ear'].y,
#             # 'right_ear_x': face['right_ear'].x,
#             # 'right_ear_y': face['right_ear'].y,
#             # 'nose_x': face['nose'].x,
#             # 'nose_y': face['nose'].y,
#             # 'mouth_x': face['mouth'].x,
#             # 'mouth_y': face['mouth'].y,
#             'left_eye_x': face['left_eye'].x - face['box'].xmin,
#             'left_eye_y': face['left_eye'].y - face['box'].ymin,
#             'right_eye_x': face['right_eye'].x - face['box'].xmin,
#             'right_eye_y': face['right_eye'].y - face['box'].ymin,
#             'left_ear_x': face['left_ear'].x - face['box'].xmin,
#             'left_ear_y': face['left_ear'].y - face['box'].ymin,
#             'right_ear_x': face['right_ear'].x - face['box'].xmin,
#             'right_ear_y': face['right_ear'].y - face['box'].ymin,
#             'nose_x': face['nose'].x - face['box'].xmin,
#             'nose_y': face['nose'].y - face['box'].ymin,
#             'mouth_x': face['mouth'].x - face['box'].xmin,
#             'mouth_y': face['mouth'].y - face['box'].ymin,
#             'color_hist_mean': face['color_hist_mean'],
#             'color_hist_std': face['color_hist_std'],
#             'color_hist_min': face['color_hist_min'],
#             'color_hist_max': face['color_hist_max'],            
#         }
#         if use_hog:
#             features.update({
#                 'hog_hist_mean': face['hog_hist_mean'],
#                 'hog_hist_std': face['hog_hist_std'],
#                 'hog_hist_min': face['hog_hist_min'],
#                 'hog_hist_max': face['hog_hist_max'],
#             })
#         return features

#     df = pd.DataFrame.from_dict([uppack_face(f['face']) for f in unpacked_faces])

#     CLUSTERING_COLS = [
#         'box_xmin',
#         'box_ymin',
#         'box_width',
#         'box_height',
#         'left_eye_x',
#         'left_eye_y',
#         'right_eye_x',
#         'right_eye_y',
#         'nose_x',
#         'nose_y',
#         'mouth_x',
#         'mouth_y',
#         # 'face_no',
#     ]
    
#     X = df[CLUSTERING_COLS].to_numpy()
#     clustering = AgglomerativeClustering().fit(X)

#     df = pd.DataFrame.from_dict([uppack_scaled_face(f['face']) for f in unpacked_faces])
#     df['frame'] = [f['frame'] for f in unpacked_faces]
#     df['label'] = clustering.labels_
#     return df



