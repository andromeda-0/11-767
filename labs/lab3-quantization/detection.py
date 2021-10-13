import torch
import torchvision
import numpy as np
from timeit import default_timer as timer


def main():
    for device_str in ["cpu"]:
        print(device_str)
        device = torch.device(device_str)
        for model in [
            torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
                    pretrained=True).to(device),
            torchvision.models.detection.ssdlite320_mobilenet_v3_large(
                    pretrained=True).to(device)
        ]:
            print(model.__class__)
            # torch.save(model, './' + str(model._get_name()) + '.pt')
            model.eval()
            x = [torch.rand(3, 224, 224, device=device)]
            for _ in range(10):
                predictions = model(x)
            # heat up

            time = np.zeros(10)
            for size in [224, 240, 260, 300, 380, 456, 528, 600]:
                x = [torch.rand(3, size, size, device=device)]
                for i in range(10):
                    start_time = timer()
                    predictions = model(x)
                    end_time = timer()
                    time[i] = end_time - start_time
                time *= 1000
                print('Not quantized. Size,', size, 'Mean:', time.mean())
                print('Standard Deviation:', time.std())

            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear})
            # torch.save(model, './' + str(model._get_name()) + '_quantized.pt')
            # print(model)

            model.eval()
            x = [torch.rand(3, 224, 224, device=device)]
            for _ in range(10):
                predictions = model(x)
            # heat up

            time = np.zeros(10)
            for size in [224, 240, 260, 300, 380, 456, 528, 600]:
                x = [torch.rand(3, size, size, device=device)]
                for i in range(10):
                    start_time = timer()
                    predictions = model(x)
                    end_time = timer()
                    time[i] = end_time - start_time
                time *= 1000
                print('Quantized Size,', size, 'Mean:', time.mean())
                print('Standard Deviation:', time.std())


if __name__ == '__main__':
    main()
