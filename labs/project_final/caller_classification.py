from classification import *


class Classify:
    def __init__(self, device, data_root='/home/zongyuez/data/FaceMask_32', epoch=8,
                 name='MobileNetV3Small_All_class_b=64lr=0.001_r32', B=64, lr=1e-3, resize=-1,
                 model_name='MobileNetV3Small_All'):
        print('Start Loading')
        params = ParamsClassification(B=B, lr=lr, verbose=False,
                                      device=device, flip=False,
                                      normalize=False,
                                      data_root=data_root,
                                      resize=-resize)
        model = eval(model_name + '(params)')
        self.learner = Learning(params, model)
        self.learner.load_model(epoch=epoch, name=name)
        self.learner._load_test()
        print('Loading Complete')

    def __call__(self):
        self.learner.infer_random_image()


if __name__ == '__main__':
    classifier = Classify('cpu')
    classifier()
