import os
import pandas as pd
from torchvision.io import read_image
from bs4 import BeautifulSoup
import os
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset


class DataConnector:
    def __init__(self, meta_dir, img_dir):
        self.meta_dir = meta_dir
        self.img_dir = img_dir
        fn = os.listdir(self.img_dir)
        self.entries = list()
        for i in range(len(fn)):
            self.entries += self._parse_meta(i)
    
    def _get_image_fn(self, image_id):
        return self._get_prefix(image_id) + ".png"

    def _get_prefix(self, image_id):
        return 'maksssksksss' + str(image_id)

    def _get_meta_fn(self, image_id):
        return self._get_prefix(image_id) + ".xml"
    
    def _parse_meta(self, image_id):
        metadata_path = os.path.join(self.meta_dir, self._get_meta_fn(image_id))
        with open(metadata_path, 'r') as f:
            data = f.read()
            soup = BeautifulSoup(data, 'lxml')
            objects = soup.find_all('object')
            res = list()
            for i, item in enumerate(objects):
                xmin = int(item.find('xmin').text)
                ymin = int(item.find('ymin').text)
                xmax = int(item.find('xmax').text)
                ymax = int(item.find('ymax').text)
                datum = dict()
                datum['image_fn'] = self._get_image_fn(image_id)
                datum['top'] = xmin
                datum['left'] = ymin
                datum['height'] = xmax - xmin
                datum['width'] = ymax - ymin
                label = -1
                label_text = item.find('name').text
                if label_text == 'with_mask':
                    label = 1
                elif label_text == 'mask_weared_incorrect':
                    label = 2
                else:
                    label = 0
                datum['label'] = label
                res.append(datum)
            return res

    def entry2df(self):
        return pd.DataFrame(self.entries)

    def triplet_split(self, balanced=True, height_thre=32, width_thre=32, test_frac=0.1, seed=200):
        df = self.entry2df()
        label_counts = df.label.value_counts()
        print(f"label counts: \n{label_counts}")
        nrows = min(label_counts.items(), key=lambda x: x[1])[1]
        if balanced:
            print(f"Balanced the data to have {nrows} rows per categories")
        train_list, dev_list, test_list = list(), list(), list()
        for l in label_counts.keys():
            if balanced:
                tmp_df = df[(df.label == l) & (df.height <= height_thre) & (df.width <= width_thre)]
                if len(tmp_df) >= nrows:
                    group_df = tmp_df
                else:
                    group_df = df[(df.label == l)]
                group_df = group_df.sample(n=nrows, random_state=seed).copy()
            else:
                group_df = df[(df.label == l)].copy()
            test_df = group_df.sample(frac=test_frac, random_state=seed)
            group_df = group_df.drop(test_df.index)
            dev_df = group_df.sample(frac=test_frac, random_state=seed)
            train_df = group_df.drop(dev_df.index)
            train_list.append(train_df)
            dev_list.append(dev_df)
            test_list.append(test_df)
        train_data = pd.concat(train_list, axis=0).sample(frac=1, random_state=seed)
        dev_data = pd.concat(dev_list, axis=0).sample(frac=1, random_state=seed)
        test_data = pd.concat(test_list, axis=0).sample(frac=1, random_state=seed)
        return train_data, dev_data, test_data



class MaskImageDataset(Dataset):
    def __init__(self, img_dir, df, height_thre=32, width_thre=32, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.entries = df
        self.h, self.w = height_thre, width_thre

    def __len__(self):
        return len(self.entries)

    def _center_crop(self, img, t, l, h, w):
        ci = t + h // 2
        cj = l + w // 2
        di, dj = self.h // 2, self.w // 2
        return img.crop((ci - di, cj - dj, ci + di, cj + dj))

    def __getitem__(self, idx):
        entry = self.entries.iloc[idx]
        img_path = os.path.join(self.img_dir, entry['image_fn'])
        t, l, h, w = entry['top'], entry['left'], entry['height'], entry['width']
        image = Image.open(img_path).convert('RGB')
        image = self._center_crop(image, t, l, h, w)
        label = entry['label']
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label