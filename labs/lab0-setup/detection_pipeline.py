import torch
import torchvision
import argparse
import cv2
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import save_image
from torchvision.io import read_image


def capture_img(img_addr='output.png'):
    def gstreamer_pipeline(capture_width=1280, capture_height=720,
                       display_width=1280, display_height=720,
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

    HEIGHT = 1280
    WIDTH = 1920
    center = (WIDTH / 2, HEIGHT / 2)
    M = cv2.getRotationMatrix2D(center, 180, 1.0)

    cam = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)


    if cam.isOpened():
        val, img = cam.read()
        if val:
            cv2.imwrite(img_addr, img)
            return True
    return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--test', default=False, action='store_true',
                         help='test mode (use dog picture to replace the camera-captured image')

    args = parser.parse_args()

    captured_img_addr = 'captured_img.png'
    dog_img_addr = 'nala.jpeg'

    success_captured = capture_img(captured_img_addr)
    assert success_captured, 'Can not capture img via camera.'

    input_img_addr = dog_img_addr if args.test else captured_img_addr
    output_img_addr = 'detected_dog_img.png' if args.test else 'detected_cap_img.png'

    input_img = torch.tensor(cv2.imread(input_img_addr), dtype=torch.float32)[None, ...].permute(0, 3, 1, 2) / 255.
    
    detector = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
    detector.eval()
    predictions = detector(input_img)

    score_threshold = 0.5
    output_img = draw_bounding_boxes(torch.tensor((input_img * 255.), dtype=torch.uint8).squeeze(), boxes=predictions[0]['boxes'][predictions[0]['scores'] > score_threshold], width=input_img.size(3) // 500)
    output_img = output_img.permute(1, 2, 0).numpy()

    cv2.imwrite(output_img_addr, output_img)