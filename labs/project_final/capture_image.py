import cv2


class Cam:
    def __init__(self, w=32, h=32, fr=60):
        self.w = w
        self.h = h
        self.fr = fr

    def gstreamer_pipeline(self, flip_method=0):
        return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            f"width=(int){self.w}, height=(int){self.h}, "
            f"format=(string)NV12, framerate=(fraction){self.fr}/1 ! "
            f"nvvidconv flip-method={flip_method} ! "
            f"video/x-raw, width=(int){self.w}, height=(int){self.h}, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
        )

    def capture_image(self):
        """
        output: (H,W,3), RGB
        """
        cam = cv2.VideoCapture(self.gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        img = None
        if cam.isOpened():
            val, img = cam.read()
        cam.release()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
