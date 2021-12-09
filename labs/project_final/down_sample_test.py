import torch
import os
import torchvision

root_dir = 'D:/11767/FaceMask/test'

down_sampled_dir = 'D:/11767/FaceMask_32/test'

resize = torchvision.transforms.Resize(32)
# applied to tensor for down-sampling behavior consistency

for dir_name in ('CMFD', 'IMFD', 'NMFD'):
    path_name = os.path.join(root_dir, dir_name, '60000')
    down_sampled_path = os.path.join(down_sampled_dir, dir_name)
    os.mkdir(down_sampled_path)
    for image_name in os.listdir(path_name):
        if image_name[0:3] != '600':
            continue
        image_path = os.path.join(path_name, image_name)
        image = torchvision.io.read_image(image_path)
        down_sampled = resize(image)
        torchvision.io.write_png(down_sampled, os.path.join(down_sampled_path, image_name))
