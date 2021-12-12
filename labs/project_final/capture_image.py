import cv2


class Cam:
    @staticmethod
    def gstreamer_pipeline(capture_width=32, capture_height=32,
                           display_width=32, display_height=32,
                           framerate=60, flip_method=0):
        return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            f"width=(int){capture_width}, height=(int){capture_height}, "
            f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
            f"nvvidconv flip-method={flip_method} ! "
            f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
        )

    @staticmethod
    def capture_image():
        """
        output: (H,W,3), RGB
        """
        cam = cv2.VideoCapture(Cam.gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        img = None
        if cam.isOpened():
            val, img = cam.read()
            # if val:
            #     cv2.imwrite(self.output, img)
        cam.release()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
