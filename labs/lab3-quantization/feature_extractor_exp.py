import numpy as np
import torch
import torchvision
from timeit import default_timer as timer
import torch.quantization
import sys
from plot import plot_line_chart


# .resnet18()  # .mobilenet_v2()
# model = torch.quantization.quantize_dynamic(original_model, {torch.nn.Linear}, dtype=torch.qint8)

def run_exp(model):
    print(f"number of model parameters: {sum([np.prod(p.size()) for p in model.parameters()])}")
    print(f"model size: {sys.getsizeof(model)}")
    # batch_size = 5
    y, x = list(), list()
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
        y.append(np.mean(dt_list))
        x.append(batch_size)
    return y, x

if __name__ == '__main__':
    model0 = torchvision.models.mobilenet_v2(pretrained=True)
    model0.eval()
    quantized_model0 = torch.quantization.quantize_dynamic(model0, {torch.nn.Linear}, dtype=torch.qint8)
    quantized_model0.eval()

    mnet_y, _ = run_exp(model0)
    qmnet_y, _ = run_exp(quantized_model0)

    model1 = torchvision.models.resnet18(pretrained=True)
    model1.eval()
    rnet_y, _ = run_exp(model1)

    quantized_model1 = torch.quantization.quantize_dynamic(model1, {torch.nn.Linear}, dtype=torch.qint8)
    quantized_model1.eval()
    qrnet_y, _ = run_exp(quantized_model1)

    # plot
    all_y = np.asarray([mnet_y, qmnet_y, rnet_y, qrnet_y])
    all_model_names = ['mobilenets_v2 original', 'mobilenets_v2 quantized', 
                       'resnet_18 original', 'resnet_18 quantized']
    plot_line_chart(all_y, all_model_names)

