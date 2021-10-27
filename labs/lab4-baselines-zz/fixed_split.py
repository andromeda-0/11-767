import argparse
import torch
import torch.nn as nn
from bs4 import BeautifulSoup
from dataclasses import dataclass, field
from torch import Tensor
import os
from torch.utils.data.dataset import T_co
from PIL import Image
from torch.utils.data import Dataset
import torchvision
from sklearn.model_selection import KFold
from typing import Sequence
from abc import ABC, abstractmethod
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from typing import List, Optional, Dict, Union
from matplotlib import pyplot as plt
from matplotlib import patches

num_workers = 4
data_root = 'D:/11767/Mask'
image_root = os.path.join(data_root, 'images')
metadata_root = os.path.join(data_root, 'annotations')


@dataclass
class Params(ABC):
    B: int = field(default=4)
    data_dir: str = field(default='')
    lr: float = field(default=1e-3)
    max_epoch: int = field(default=101)
    is_double: int = field(default=False)
    device: torch.device = field(default=torch.device("cuda:0"))
    dropout: float = field(default=0.)
    input_dims: tuple = field(default=(3, 1024, 1024))
    output_channels: int = field(default=3)

    @abstractmethod
    def __str__(self):
        return ''


def collate_fn(batch):
    return tuple(zip(*batch))


def plot_image(img_tensor, annotation):
    fig, ax = plt.subplots(1)
    img = img_tensor.cpu().data

    # Display the image
    ax.imshow(img.permute(1, 2, 0))

    for score, box in zip(annotation['scores'], annotation["boxes"]):
        if score < 0.5:
            continue
        xmin, ymin, xmax, ymax = box

        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1,
                                 edgecolor='r', facecolor='none')

        ax.add_patch(rect)

    plt.show()


def load_label(image_id: int) -> Dict:
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
        N = len(objects)
        boxes = torch.zeros((N, 4), dtype=torch.float)
        labels = torch.zeros(N, dtype=torch.long)

        for i, item in enumerate(objects):
            boxes[i, 0] = int(item.find('xmin').text)
            boxes[i, 1] = int(item.find('ymin').text)
            boxes[i, 2] = int(item.find('xmax').text)
            boxes[i, 3] = int(item.find('ymax').text)

            label_text = item.find('name').text
            if label_text == 'with_mask':
                labels[i] = 1
            elif label_text == 'mask_weared_incorrect':
                labels[i] = 2
            else:
                labels[i] = 0
        return {'boxes': boxes, 'labels': labels}


class DetectionSet(Dataset):
    def __getitem__(self, index) -> T_co:
        """
        Image, Label
        """
        image_path = 'maksssksksss' + str(index) + '.png'
        image = Image.open(os.path.join(image_root, image_path)).convert('RGB')
        label = load_label(index)

        return image, label

    def __len__(self):
        return self.size

    def __init__(self):
        self.size = len(os.listdir(image_root))


class Subset(Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int], transforms) -> None:
        self.transforms = transforms
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        item = self.dataset[self.indices[idx]]
        return self.transforms(item[0]), item[1]

    def __len__(self):
        return len(self.indices)


class ParamsDetection(Params):
    def __init__(self, B, lr, dropout, device, flip, normalize, rotate,
                 erase, perspective, max_epoch=201, data_dir='D:/11767/Mask'):

        self.size = 1024

        super().__init__(B=B, lr=lr, max_epoch=max_epoch, dropout=dropout, output_channels=4000,
                         data_dir=data_dir, device=device, input_dims=(3, self.size, self.size))

        self.str = 'class_b=' + str(self.B) + 'lr=' + str(
                self.lr) + '_'  # 'd=' + str(self.dropout)'

        transforms_train = []
        transforms_test = []

        if flip:
            transforms_train.append(torchvision.transforms.RandomHorizontalFlip())
            self.str = self.str + 'f'

        if rotate:
            transforms_train.append(torchvision.transforms.RandomRotation(15))
            self.str = self.str + 'r'

        if perspective:
            transforms_train.append(torchvision.transforms.RandomPerspective())
            self.str = self.str + 'p'

        transforms_train.append(torchvision.transforms.ToTensor())
        transforms_test.append(torchvision.transforms.ToTensor())

        if normalize:
            self.str = self.str + 'n'
            transforms_test.append(
                    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
            transforms_train.append(
                    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))

        if erase:
            transforms_train.append(torchvision.transforms.RandomErasing())
            self.str = self.str + 'e'

        self.transforms_train = torchvision.transforms.Compose(transforms_train)
        self.transforms_test = torchvision.transforms.Compose(transforms_test)

    def __str__(self):
        return self.str


class Model(nn.Module, ABC):
    @abstractmethod
    def __init__(self, params: ParamsDetection):
        super().__init__()
        self.params = params

    @abstractmethod
    def forward(self, images: List[Tensor], annotations: Optional[List[Dict]] = None) -> Union[
        Dict, List[Dict]]:
        pass

    @abstractmethod
    def trainable(self):
        pass


class FasterRCNN_mobilenet_v3_large_fpn(Model):
    def __init__(self, params: ParamsDetection):
        super(FasterRCNN_mobilenet_v3_large_fpn, self).__init__(params)
        self.net = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
                pretrained=True)
        in_features = self.net.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.net.roi_heads.box_predictor = \
            torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                    in_features, self.params.output_channels)

    def forward(self, images: List[Tensor], annotations: Optional[List[Dict]] = None) -> Union[
        Dict, List[Dict]]:
        return self.net(images, annotations)

    def trainable(self):
        return self.net.roi_heads.box_predictor.parameters()


class Learning(ABC):
    def __init__(self, params: ParamsDetection, model: Model, optimizer_handle=torch.optim.Adam,
                 criterion_handle=None, draw_graph=False, string=None):

        self.params: ParamsDetection = params
        self.device = params.device
        self.str = model.__class__.__name__ + '_' + str(params) if string is None else string

        self.writer = SummaryWriter('runs/' + str(self))

        self.model = model.cuda(self.device)
        if params.is_double:
            self.model.double()

        if draw_graph:
            self.writer.add_graph(model, torch.rand([params.B] + list(params.input_dims),
                                                    device=self.device))
        if optimizer_handle is not None:
            self.optimizer = optimizer_handle(self.model.trainable(), lr=self.params.lr)
        else:
            self.optimizer = None

        if criterion_handle is not None:
            self.criterion = criterion_handle().cuda(self.device)
        else:
            self.criterion = None

        self.init_epoch = 0

        self.k_fold = KFold(n_splits=5, shuffle=True)

        self.dataset = DetectionSet()
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

        self.label_to_class = {0: 'without_mask', 1: 'with_mask', 2: 'mask_weared_incorrect'}

        self.train_split = list(range(700))
        self.test_split = list(range(700, 800))
        self.valid_split = list(range(800, 853))

    def __del__(self):
        self.writer.flush()
        self.writer.close()

    def __str__(self):
        return self.str

    def _load_train(self):
        train_set = Subset(self.dataset, self.train_split, transforms=self.params.transforms_train)
        self.train_loader = torch.utils.data.DataLoader(train_set,
                                                        batch_size=self.params.B, shuffle=True,
                                                        pin_memory=True, num_workers=num_workers,
                                                        collate_fn=collate_fn)

    def _load_valid(self):
        valid_set = Subset(self.dataset, self.valid_split, transforms=self.params.transforms_test)
        self.valid_loader = torch.utils.data.DataLoader(valid_set,
                                                        batch_size=self.params.B, shuffle=False,
                                                        pin_memory=True, num_workers=num_workers,
                                                        collate_fn=collate_fn)

    def _load_test(self):
        test_set = Subset(self.dataset, self.test_split, transforms=self.params.transforms_test)
        self.test_loader = torch.utils.data.DataLoader(test_set,
                                                       batch_size=self.params.B, shuffle=False,
                                                       pin_memory=True, num_workers=num_workers,
                                                       collate_fn=collate_fn)

    def load_model(self, epoch=20, name=None, model=True, optimizer=True, loss=False):
        if name is None:
            loaded = torch.load('checkpoints/' + str(self) + 'e=' + str(epoch) + '.pt',
                                map_location=self.device)
        else:
            loaded = torch.load('checkpoints/' + name + 'e=' + str(epoch) + '.pt',
                                map_location=self.device)
        self.init_epoch = loaded['epoch']

        if model:
            self.model.load_state_dict(loaded['model_state_dict'])
        if optimizer:
            self.optimizer.load_state_dict(loaded['optimizer_state_dict'])
        if loss:
            if 'loss_state_dict' in loaded:
                self.criterion.load_state_dict(loaded['loss_state_dict'], strict=False)

    def save_model(self, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_state_dict': self.criterion.state_dict() if self.criterion is not None else None,
        }, 'checkpoints/' + str(self) + 'e=' + str(epoch) + '.pt')

    def train(self, checkpoint_interval=20):
        if self.train_loader is None:
            self._load_train()

        self._validate(0)

        print('Training...')
        with torch.cuda.device(self.device):
            self.model.train()
            for epoch in range(self.init_epoch + 1, self.params.max_epoch):
                total_loss = torch.zeros(1, device=self.device)
                for i, batch in enumerate(tqdm(self.train_loader)):
                    images = list(bx.to(self.device) for bx in batch[0])
                    annotations = list(
                            {k: v.to(self.device) for k, v in by.items()} for by in batch[1])

                    loss_dict = self.model(images, annotations)

                    loss = torch.stack([l for l in loss_dict.values()]).sum()

                    total_loss += loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                loss_item = total_loss.item() / (i + 1)
                self.writer.add_scalar('Loss/Train', loss_item, epoch)
                print('epoch: ', epoch, 'Training Loss: ', "%.5f" % loss_item)

                self._validate(epoch)
                self.model.train()

                if epoch % checkpoint_interval == 0:
                    self.save_model(epoch)

    def test(self):
        pass

    def _validate(self, epoch, verbose=True):
        if self.valid_loader is None:
            self._load_valid()

        # print('Validating...')
        with torch.cuda.device(self.device):
            with torch.no_grad():
                self.model.eval()
                total_acc = torch.zeros(1, device=self.device)
                for i, batch in enumerate(self.valid_loader):
                    images = list(bx.to(self.device) for bx in batch[0])
                    annotations = list(
                            {k: v.to(self.device) for k, v in by.items()} for by in batch[1])
                    output = self.model(images)
                    if verbose:
                        plot_image(images[0], output[0])

        #             prediction = list(result['labels'] for result in output)
        #             y_prime = torch.argmax(prediction, dim=1)
        #             total_acc += torch.count_nonzero(torch.eq(y_prime, annotations))
        #
        #         accuracy_item = total_acc.item() / (i + 1) / self.params.B
        #         self.writer.add_scalar('Accuracy/Validation', accuracy_item, epoch)
        #         print('epoch: ', epoch, 'Validation Accuracy: ', "%.5f" % accuracy_item)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help='Batch Size', default=1, type=int)
    parser.add_argument('--dropout', default=0.4, type=float)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--gpu_id', help='GPU ID (0/1)', default='0')
    parser.add_argument('--model', default='FasterRCNN_mobilenet_v3_large_fpn', help='Model Name')
    parser.add_argument('--epoch', default=-1, help='Load Epoch', type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--erase', action='store_true')
    parser.add_argument('--save', default=10, type=int, help='Checkpoint interval')
    parser.add_argument('--perspective', action='store_true')
    parser.add_argument('--load', default='', help='Load Name')
    parser.add_argument('--rotate', action='store_true')

    args = parser.parse_args()

    params = ParamsDetection(B=args.batch, dropout=args.dropout, lr=args.lr,
                             device='cuda:' + args.gpu_id, flip=args.flip,
                             normalize=args.normalize, erase=args.erase,
                             perspective=args.perspective,
                             rotate=args.rotate)
    model = eval(args.model + '(params)')
    learner = Learning(params, model)
    if args.epoch >= 0:
        if args.load == '':
            learner.load_model(args.epoch)
        else:
            learner.load_model(args.epoch, args.load)

    if args.train:
        learner.train()
    if args.test:
        learner.test()


if __name__ == '__main__':
    main()
