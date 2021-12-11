from classification import *


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


def mask_detection_caller():
    from timeit import default_timer as timer
    t0 = timer()
    classifier_instance()
    return (timer() - t0) * 1000


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vision_device', default='cpu')
    parser.add_argument('--vision_weights_name',
                        default='MobileNetV3Small_All_class_b=64lr=0.001_r32')
    parser.add_argument('--vision_model_name', default='MobileNetV3Small_All')
    parser.add_argument('--resize', type=int, default=-1)
    parser.add_argument('--epoch', type=int, default=8)
    parser.add_argument('--image_root', default='/home/zongyuez/data/FaceMask_32')

    args = parser.parse_args()

    classifier_instance = Classify(device=args.vision_device, name=args.vision_weights_name,
                                   model_name=args.vision_model_name, resize=args.resize,
                                   epoch=args.epoch, data_root=args.image_root)

    print('Time Used: %.2f' % (mask_detection_caller()))
