Lab 0: Hardware setup
===
The goal of this lab is for you to become more familiar with the hardware platform you will be working with this semester, and for you to complete basic setup so that everyone in the group should be able to work remotely on the device going forward. By the end of class today, everyone in your group should be able to ssh in to the device, use the camera to take a picture, record audio, run a basic NLP model, and run a basic CV model. 

If you successfully complete all those tasks, then your final task is to write a script that pipes together I/O with a model. For example, you could write a script that uses the camera to capture an image, then runs classification on that image. Or you could capture audio, run speech-to-text, then run sentiment analysis on that text.

Group name: tripleFighting
---
Group members present in lab today: Jiajun Bao, Songhao Jia, Zongyue Zhao.

1: Set up your device.
----
Depending on your hardware, follow the instructions provided in this directory: [Raspberry Pi 4](https://github.com/strubell/11-767/blob/main/labs/lab0-setup/setup-rpi4.md), [Jetson Nano](https://github.com/strubell/11-767/blob/main/labs/lab0-setup/setup-jetson.md), [Google Coral](https://coral.ai/docs/dev-board/get-started/). 
1. What device(s) are you setting up? \
  a. Jetson Nano 2GB.
3. Did you run into any roadblocks following the instructions? What happened, and what did you do to fix the problem?\
  a. We could not find the `/boot/firmware` folder. Solution: manually creating the folder and the corresponding config file. (No longer needed).\
  b. `python3 capture_audio.py` generates wav file with no content. Debug process: ensure input device is set correctly (USB), check numpy array 
3. Are all group members now able to ssh in to the device from their laptops? If not, why not? How will this be resolved?
 a. Yes.

2: Collaboration / hardware management plan
----
4. What is your group's hardware management plan? For example: Where will the device(s) be stored throughout the semester? What will happen if a device needs physical restart or debugging? What will happen in the case of COVID lockdown?\
  a. Our group has purchased an extra Jetson Nano 2GB, which is now stored in Professor Strubell's office. All group members has ssh access to it, and we will contact Prof. Strubell for rebooting if needed. In case of a COVID lockdown, we will use the other Jetson board.

3: Putting it all together
----
5. Now, you should be able to take a picture, record audio, run a basic computer vision model, and run a basic NLP model. Now, write a script that pipes I/O to models. For example, write a script that takes a picture then runs a detection model on that image, and/or write a script that . Include the script at the end of your lab report.\
  a. Code attached to the end.
6. Describe what the script you wrote does (document it.) \
In our script, we run a detection model on the picture we captured or an arbitrary image. We first capture a picture with the camera, and store it into the disk. After that, we either load the captured image or an arbitrary image we set to run the detection model. Finally we store the detection result as an image into the disk.
7. Did you have any trouble getting this running? If so, describe what difficulties you ran into, and how you tried to resolve them. \
One of the trouble was installing torchvision package. Since nvidia jetson uses ARM architecture, there was no way to install torchvision in normal way. Therefore we had to download the source code, and ran & compiled it manually.
Another trouble we encountered was that pytorch or torchvision always came out abnormal errors, expecially during IO. I believed the reason would be  some instructions used to load and store were not capable with ARM processor. I changed all IO functions into opencv functions to solve the problem.
```
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
```
