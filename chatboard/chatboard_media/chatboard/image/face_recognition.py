
from insightface.app import FaceAnalysis
from config import AWS_IMAGE_BUCKET, FACE_DETECTION_MODEL, BASE_DIR, INSIGHT_FACE_MODEL_DIR




class FaceRecognizer:

    def __init__(self) -> None:
        self.app = FaceAnalysis(name=FACE_DETECTION_MODEL, root=str(INSIGHT_FACE_MODEL_DIR), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640,640))


    def recognize(self, image):
        faces = self.app.get(image)
        return faces