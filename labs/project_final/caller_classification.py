from classification import *


class Classify:
    def __init__(self):
        print('Start Loading')
        params = ParamsClassification(B=64, lr=1e-3, verbose=False,
                                      device='cuda:0', flip=False,
                                      normalize=False,
                                      data_root='/home/zongyuez/data/FaceMask',
                                      resize=224)
        model = MobileNetV3Small_All(params)
        self.learner = Learning(params, model)
        self.learner.load_model(epoch=2)
        print('Loading Complete')

    def __call__(self):
        self.learner.test()


if __name__ == '__main__':
    classifier = Classify()
    classifier()
