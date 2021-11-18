import numpy as np
from bs4 import BeautifulSoup
from dataclasses import dataclass, field
from torch import Tensor
import os

data_root = '/home/zongyuez/data/Mask'
image_root = os.path.join(data_root, 'images')
metadata_root = os.path.join(data_root, 'annotations')
label_root = os.path.join(data_root, 'classification_labels.npy')


def load_label(image_id: int) -> int:
    """
    integer 0 (no_mask), 1 (correct_mask), or 2 (incorrect_mask).
    """
    metadata_path = os.path.join(metadata_root, 'maksssksksss' + str(image_id) + '.xml')
    with open(metadata_path, 'r') as f:
        data = f.read()
        soup = BeautifulSoup(data, 'lxml')
        objects = soup.find_all('object')

        has_correct_mask = False
        has_incorrect_mask = False

        for i, item in enumerate(objects):
            label_text = item.find('name').text
            if label_text == 'with_mask':
                has_correct_mask = True
            elif label_text == 'mask_weared_incorrect':
                has_incorrect_mask = True

        if has_incorrect_mask:
            return 2

        if has_correct_mask:
            return 1

        return 0


def main():
    N = 853
    labels = np.zeros(N, dtype=np.long)

    for i in range(N):
        labels[i] = load_label(i)

    np.save(label_root, labels)


if __name__ == '__main__':
    main()
