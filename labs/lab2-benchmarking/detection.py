import torch
import torchvision
import numpy as np
from timeit import default_timer as timer


def main():
    for device_str in ["cuda:0", "cpu"]:
        print(device_str)
        device = torch.device(device_str)
        for model in [torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
                pretrained=True).to(device),
                      torchvision.models.detection.ssdlite320_mobilenet_v3_large(
                              pretrained=True).to(device)]:
            print(model.__class__)
            num_params = sum([np.prod(p.size()) for p in model.parameters()])
            print(num_params)

            model.eval()
            x = [torch.rand(3, 224, 224, device=device)]
            for _ in range(10):
                predictions = model(x)
            # heat up

            time = np.zeros(10)
            # for size in [224, 240, 260, 300, 380, 456, 528, 600]:
            for size in [224]:
                x = [torch.rand(3, size, size, device=device)]
                for i in range(10):
                    start_time = timer()
                    predictions = model(x)
                    end_time = timer()
                    time[i] = end_time - start_time

                print('Size,', size, 'Mean:', time.mean())
                print('Standard Deviation:', time.std())


if __name__ == '__main__':
    main()

# ssdlite320_mobilenet_v3_large
# Size, 224 Mean: 0.3020110395984375
# Standard Deviation: 0.005912020144749417
# Size, 240 Mean: 0.29990467650350183
# Standard Deviation: 0.005574685380426938
# Size, 260 Mean: 0.29828336329956073
# Standard Deviation: 0.003632252631160885
# Size, 300 Mean: 0.2976852484003757
# Standard Deviation: 0.001640473373306994
# Size, 380 Mean: 0.28501639430178327
# Standard Deviation: 0.0014071642840417006
# Size, 456 Mean: 0.29033255520043894
# Standard Deviation: 0.0019010920843747748
# Size, 528 Mean: 0.2927533374997438
# Standard Deviation: 0.009867954336565574
# Size, 600 Mean: 0.28535580430034313
# Standard Deviation: 0.0015126524799162593

# faster rcnn
# Size, 224 Mean: 0.09817643930000486
# Standard Deviation: 0.005144447013551248
# Size, 240 Mean: 0.10615955569810467
# Standard Deviation: 0.007121825903890122
# Size, 260 Mean: 0.10187393239903031
# Standard Deviation: 0.007719714949921088
# Size, 300 Mean: 0.11734106359508586
# Standard Deviation: 0.012886537295055024
# Size, 380 Mean: 0.12151471519755433
# Standard Deviation: 0.00841067258603119
# Size, 456 Mean: 0.1079640667012427
# Standard Deviation: 0.004974017054939631
# Size, 528 Mean: 0.10361214489967097
# Standard Deviation: 0.006131127774616639
# Size, 600 Mean: 0.1068273856988526
# Standard Deviation: 0.0009570182228357254

# on desktop gpu/cpu
# cuda:0
# <class 'torchvision.models.detection.faster_rcnn.FasterRCNN'>
# 19386354
# Size, 224 Mean: 0.01445891167732212
# Standard Deviation: 5.880855513996412e-05
# <class 'torchvision.models.detection.ssd.SSD'>
# 3440060
# Size, 224 Mean: 0.0719202999263873
# Standard Deviation: 0.00041043049684485447
# cpu
# <class 'torchvision.models.detection.faster_rcnn.FasterRCNN'>
# 19386354
# Size, 224 Mean: 0.05502555492387997
# Standard Deviation: 0.00199144965521597
# <class 'torchvision.models.detection.ssd.SSD'>
# 3440060
# Size, 224 Mean: 0.07409234972601268
# Standard Deviation: 0.0023820226879700908
