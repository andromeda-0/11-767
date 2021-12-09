import torch
import os
import torchvision

root_dir = 'C:/Users/Zongyue/Downloads/In-the-wild/test'

down_sampled_dir = 'C:/Users/Zongyue/Downloads/In-the-wild-224/test'

resize = torchvision.transforms.Resize(224)
# applied to tensor for down-sampling behavior consistency

for dir_name in ('CMFD', 'IMFD', 'NMFD'):
    path_name = os.path.join(root_dir, dir_name)
    down_sampled_path = os.path.join(down_sampled_dir, dir_name)
    os.mkdir(down_sampled_path)
    for image_name in os.listdir(path_name):
        image_path = os.path.join(path_name, image_name)
        image = torchvision.io.read_image(image_path)
        down_sampled = resize(image)
        torchvision.io.write_png(down_sampled, os.path.join(down_sampled_path, image_name))
