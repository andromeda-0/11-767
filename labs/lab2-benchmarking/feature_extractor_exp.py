import numpy as np
import torch
import torchvision
from timeit import default_timer as timer


def main():
    model = torchvision.models.mobilenet_v2()  # .resnet18()  # .mobilenet_v2()
    model.eval()
    print(f"number of model parameters: {sum([np.prod(p.size()) for p in model.parameters()])}")

    # batch_size = 5
    for batch_size in [1, 1, 2, 4, 8, 16, 32]:
        dt_list = list()
        num_trials = 10
        for _ in range(num_trials):
            start_time = timer()
            input_batch = torch.rand((batch_size, 3, 224, 224))
            
            if torch.cuda.is_available():
                input_batch = input_batch.to("cuda")
                model.to('cuda')
            # start_time = timer()
            with torch.no_grad():
                output = model(input_batch)
            
            end_time = timer()

            dt = end_time - start_time
            dt_list.append(dt / batch_size) 
        print(f"bs: {batch_size} average time: {np.mean(dt_list)} (±{np.std(dt_list)})")

main()
