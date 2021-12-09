from classification import *


class Classify:
    def __init__(self):
        print('Start Loading')
        params = ParamsClassification(B=64, lr=1e-3, verbose=False,
                                      device='cpu', flip=False,
                                      normalize=False,
                                      data_root='/home/zongyuez/data/In-the-wild-224',
                                      resize=32)
        model = MobileNetV3Small_All(params)
        self.learner = Learning(params, model)
        self.learner.load_model(epoch=8, name='MobileNetV3Small_All_class_b=64lr=0.001_r32')
        self.learner._load_test()
        print('Loading Complete')

    def __call__(self):
        self.learner.infer_random_image()


if __name__ == '__main__':
    classifier = Classify()
    classifier()
