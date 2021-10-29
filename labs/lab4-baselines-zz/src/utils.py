from bs4 import BeautifulSoup
import os
import torch


def get_image_crop_meta(metadata_root: str, image_id: int):
    """
    dict{
    'boxes':FloatTensor[N, 4]: [x1, y1, x2, y2], 0 <= x1 < x2 <= W, 0 <= y1 < y2 <= H.
    'labels':Int64Tensor[N]
    }
    """
    metadata_path = os.path.join(metadata_root, 'maksssksksss' + str(image_id) + '.xml')
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
            datum['image_id'] = image_id
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


