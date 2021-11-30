import argparse

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
import os
from torch.utils.data.dataset import T_co
from PIL import Image
from torch.utils.data import Dataset
import torchvision
from typing import Sequence
from abc import ABC, abstractmethod
from tqdm import tqdm
from torchmetrics.functional import iou
from timeit import default_timer as timer

try:
    from matplotlib import pyplot as plt
    from matplotlib import patches
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    plt = None
    patches = None
    SummaryWriter = None


@dataclass
class Params(ABC):
    B: int = field(default=4)
    lr: float = field(default=1e-3)
    max_epoch: int = field(default=101)
    is_double: int = field(default=False)
    device: torch.device = field(default=torch.device("cuda:0"))
    input_dims: tuple = field(default=(3, 1024, 1024))
    output_channels: int = field(default=3)

    @abstractmethod
    def __str__(self):
        return ''


class ClassificationSet(Dataset):
    def __getitem__(self, index) -> T_co:
        """
        Image, Label
        """
        image_path = 'maksssksksss' + str(index) + '.png'
        image = Image.open(os.path.join(self.image_root, image_path)).convert('RGB')
        label = self.labels[index]

        return image, label

    def __len__(self):
        return self.size

    def __init__(self, image_root):
        self.image_root = image_root
        self.size = len(os.listdir(image_root))
        self.labels = np.load(label_root)


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


class ParamsClassification(Params):
    def __init__(self, B, lr, device, flip, normalize, vis_threshold, verbose,
                 max_epoch=101):

        super().__init__(B=B, lr=lr, max_epoch=max_epoch, output_channels=3,
                         device=device, input_dims=(3, 480, 640))

        self.vis_threshold = vis_threshold
        self.str = 'class_b=' + str(self.B) + 'lr=' + str(self.lr) + '_'
        self.verbose = verbose

        transforms_train = []
        transforms_test = []

        if flip:
            transforms_train.append(torchvision.transforms.RandomHorizontalFlip())
            self.str = self.str + 'f'

        transforms_train.append(torchvision.transforms.ToTensor())
        transforms_test.append(torchvision.transforms.ToTensor())

        if normalize:
            self.str = self.str + 'n'
            transforms_test.append(
                    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
            transforms_train.append(
                    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))

        self.transforms_train = torchvision.transforms.Compose(transforms_train)
        self.transforms_test = torchvision.transforms.Compose(transforms_test)

    def __str__(self):
        return self.str


class Model(nn.Module, ABC):
    @abstractmethod
    def __init__(self, params: ParamsClassification):
        super().__init__()
        self.params = params

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def trainable(self):
        pass


class MobileNetV3Large(Model):
    def __init__(self, params: ParamsClassification):
        super().__init__(params)
        self.net = torchvision.models.mobilenet_v3_large(pretrained=True, progress=False).features

        for param in self.net.parameters():
            param.requires_grad = False

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(960, params.output_channels)

    def forward(self, x):
        x = self.net(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def trainable(self):
        return self.classifier.parameters()


class Learning(ABC):
    def __init__(self, params: ParamsClassification, model: Model,
                 optimizer_handle=torch.optim.Adam,
                 criterion_handle=nn.CrossEntropyLoss, draw_graph=False, string=None):

        self.params: ParamsClassification = params
        self.device = params.device
        self.str = model.__class__.__name__ + '_' + str(params) if string is None else string

        self.writer = SummaryWriter('runs/' + str(self)) if SummaryWriter is not None else None

        self.model = model.cuda(self.device)
        if params.is_double:
            self.model.double()

        if draw_graph and SummaryWriter is not None:
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

        self.dataset = ClassificationSet(image_root)
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

        self.label_to_class = {0: 'without_mask', 1: 'with_mask', 2: 'mask_weared_incorrect'}

        self.train_split = list(range(700))
        self.test_split = list(range(700, 800))
        self.valid_split = list(range(800, 853))

    def __del__(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

    def __str__(self):
        return self.str

    def _load_train(self):
        train_set = Subset(self.dataset, self.train_split, transforms=self.params.transforms_train)
        self.train_loader = torch.utils.data.DataLoader(train_set,
                                                        batch_size=self.params.B, shuffle=True,
                                                        pin_memory=True, num_workers=num_workers)

    def _load_valid(self):
        valid_set = Subset(self.dataset, self.valid_split, transforms=self.params.transforms_test)
        self.valid_loader = torch.utils.data.DataLoader(valid_set,
                                                        batch_size=self.params.B, shuffle=False,
                                                        pin_memory=True, num_workers=num_workers)

    def _load_test(self):
        test_set = Subset(self.dataset, self.test_split, transforms=self.params.transforms_test)
        self.test_loader = torch.utils.data.DataLoader(test_set,
                                                       batch_size=self.params.B, shuffle=False,
                                                       pin_memory=True, num_workers=num_workers)

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
        self.model.train()
        for epoch in range(self.init_epoch + 1, self.params.max_epoch):
            total_loss = torch.zeros(1, device=self.device)
            total_acc = torch.zeros(1, device=self.device)
            i = 0
            for i, batch in enumerate(tqdm(self.train_loader)):
                bx = batch[0].to(self.device)
                by = batch[1].to(self.device)

                prediction = self.model(bx)
                loss = self.criterion(prediction, by)

                total_loss += loss
                y_prime = torch.argmax(prediction, dim=1)
                total_acc += torch.count_nonzero(torch.eq(y_prime, by))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            loss_item = total_loss.item() / (i + 1)
            accuracy_item = total_acc.item() / (i + 1) / self.params.B

            if self.writer is not None:
                self.writer.add_scalar('Loss/Train', loss_item, epoch)
                self.writer.add_scalar('Accuracy/Train', accuracy_item, epoch)
            print('epoch: ', epoch, 'Training Loss: ', "%.5f" % loss_item,
                  'Accuracy: ', "%.5f" % accuracy_item)

            self._validate(epoch)
            self.model.train()

            if epoch % checkpoint_interval == 0:
                self.save_model(epoch)

    def test(self):
        self._validate(self.init_epoch, mode='Test')

    def _validate(self, epoch, mode='Validation'):
        if mode == 'Validation':
            if self.valid_loader is None:
                self._load_valid()
            loader = self.valid_loader
        else:
            if self.test_loader is None:
                self._load_test()
            loader = self.test_loader

        with torch.no_grad():
            start_time = timer()

            self.model.eval()
            total_loss = torch.zeros(1, device=self.device)
            total_acc = torch.zeros(1, device=self.device)

            for i, batch in enumerate(loader):
                bx = batch[0].to(self.device)
                by = batch[1].to(self.device)

                prediction = self.model(bx)
                loss = self.criterion(prediction, by)
                total_loss += loss
                y_prime = torch.argmax(prediction, dim=1)
                total_acc += torch.count_nonzero(torch.eq(y_prime, by))

            end_time = timer()

            time_per_iter = (end_time - start_time) // i // self.params.B

            loss_item = total_loss.item() / (i + 1)
            accuracy_item = total_acc.item() / (i + 1) / self.params.B
            iou_item = iou(prediction, by, num_classes=self.params.output_channels)
            if self.writer is not None:
                self.writer.add_scalar('Loss/' + mode, loss_item, epoch)
                self.writer.add_scalar('Accuracy/' + mode, accuracy_item, epoch)
                self.writer.add_scalar('mIoU/' + mode, iou_item, epoch)
                self.writer.add_scalar('Latency/' + mode, time_per_iter, epoch)
            print('epoch: ', epoch, mode + ' Loss: ', "%.5f" % loss_item,
                  'Accuracy: ', "%.5f" % accuracy_item, 'IoU: ', "%.5f" % iou_item,
                  'Latency: ', "%.5f" % time_per_iter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help='Batch Size', default=1, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--gpu_id', help='GPU ID (0/1)', default='0')
    parser.add_argument('--model', default='MobileNetV3Large', help='Model Name')
    parser.add_argument('--epoch', default=-1, help='Load Epoch', type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--save', default=2, type=int, help='Checkpoint interval')
    parser.add_argument('--load', default='', help='Load Name')
    parser.add_argument('--vis_threshold', default=0.0, type=float)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--data_root', default='/home/zongyuez/data/Mask')

    args = parser.parse_args()

    num_workers = 4
    data_root = args.data_root
    image_root = os.path.join(data_root, 'images')
    metadata_root = os.path.join(data_root, 'annotations')
    label_root = os.path.join(data_root, 'classification_labels.npy')

    params = ParamsClassification(B=args.batch, lr=args.lr, verbose=args.verbose,
                                  device='cuda:' + args.gpu_id, flip=args.flip,
                                  normalize=args.normalize, vis_threshold=args.vis_threshold)
    model = eval(args.model + '(params)')
    learner = Learning(params, model)
    if args.epoch >= 0:
        if args.load == '':
            learner.load_model(args.epoch)
        else:
            learner.load_model(args.epoch, args.load)

    if args.train:
        learner.train(args.save)
    if args.test:
        learner.test()
