import time

import torchvision.io

from classification import *
from capture_image import *
import cv2


class Classify:
    def __init__(self, device, data_root='/home/zongyuez/data/FaceMask_32', epoch=8,
                 name='MobileNetV3Small_All_class_b=64lr=0.001_r32', resize=-1,
                 model_name='MobileNetV3Small_All'):
        print('Start Loading')
        params = ParamsClassification(B=1, lr=1e-3, verbose=False,
                                      device=device, flip=False,
                                      normalize=False,
                                      data_root=data_root,
                                      resize=resize)
        model = eval(model_name + '(params)')
        self.learner = Learning(params, model)
        self.learner.load_model(epoch=epoch, name=name)
        self.learner._load_test()
        print('Loading Complete')

    def __call__(self):
        self.learner.infer_random_image()


class Classify_Camera:
    def __init__(self, device, data_root='/home/zongyuez/data/FaceMask_32', epoch=8,
                 name='MobileNetV3Small_All_class_b=64lr=0.001_r32', resize=-1,
                 model_name='MobileNetV3Small_All', normalize=False):
        print('Start Loading')
        params = ParamsClassification(B=1, lr=1e-3, verbose=False,
                                      device=device, flip=False,
                                      normalize=normalize,
                                      data_root=data_root,
                                      resize=resize)
        model = eval(model_name + '(params)')
        self.learner = Learning(params, model)
        self.learner.load_model(epoch=epoch, name=name)
        self.learner._load_test()  # needed for classes annotation
        self.resize = resize
        self.normalize = normalize
        print('Loading Complete')

    def __call__(self, img):
        self.learner.infer_camera_image(img, resize=self.resize, normalize=self.normalize)


def mask_detection_caller(cam):
    from timeit import default_timer as timer
    t0 = timer()
    img = cam.capture_image()
    t1 = timer()
    classifier_instance(img)
    t2 = timer()
    return (t2 - t1) * 1000, (t1 - t0) * 1000, img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vision_device', default='cpu')
    parser.add_argument('--vision_weights_name',
                        default='MobileNetV3Small_All_class_b=64lr=0.001_nr32')
    parser.add_argument('--vision_model_name', default='MobileNetV3Small_All')
    parser.add_argument('--resize', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=9)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--input_image', default='')

    args = parser.parse_args()

    classifier_instance = Classify_Camera(device=args.vision_device, name=args.vision_weights_name,
                                          model_name=args.vision_model_name, resize=args.resize,
                                          epoch=args.epoch, normalize=args.normalize)

    if args.input_image == '':
        cam = Cam_Always_On()
        t_p = 0
        t_c = 0
        for _ in range(10):
            t_prediction, t_camera, img = mask_detection_caller(cam)
            t_p += t_prediction
            t_c += t_camera
            time.sleep(10)
        print('Time Used by Prediction: %.1f ms, Time Used by Camera: %.1f ms' % (
            t_p / 10, t_c / 10))

    else:
        # image = cv2.imread('/home/zongyuez/data/In-the-wild-224/test/CMFD/20211208_221423.jpg')
        image = cv2.imread(args.input_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        classifier_instance(image)
