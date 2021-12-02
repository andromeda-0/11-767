import argparse

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
import os
from torch.utils.data.dataset import T_co
import torchvision
from abc import ABC, abstractmethod

from torchvision.datasets import ImageFolder
from tqdm import tqdm
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
    B: int = field()
    lr: float = field()
    max_epoch: int = field()
    device: torch.device = field()
    input_dims: tuple = field()
    output_channels: int = field()
    is_double: int = field(default=False)

    @abstractmethod
    def __str__(self):
        return ''


class FaceMaskSet(ImageFolder):
    def __init__(self, image_root, transforms):
        super(FaceMaskSet, self).__init__(root=image_root, transform=transforms)


class ParamsClassification(Params):
    def __init__(self, B, lr, device, flip, normalize, verbose, resize,
                 max_epoch=101, data_root='D:/11767/FaceMask'):

        self.size = resize if resize > 0 else 1024
        super().__init__(B=B, lr=lr, max_epoch=max_epoch, output_channels=3,
                         device=device, input_dims=(3, self.size, self.size))

        self.str = 'class_b=' + str(self.B) + 'lr=' + str(self.lr) + '_'
        self.verbose = verbose
        self.data_root = data_root

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

        if resize > 0:
            self.str = self.str + 'r' + str(resize)
            transforms_train.append(torchvision.transforms.Resize(resize))
            transforms_test.append(torchvision.transforms.Resize(resize))

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


class MobileNetV3Small(Model):
    def __init__(self, params: ParamsClassification):
        super().__init__(params)
        self.net = torchvision.models.mobilenet_v3_small(pretrained=True, progress=False).features

        for param in self.net.parameters():
            param.requires_grad = False

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(576, params.output_channels)

    def forward(self, x):
        x = self.net(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def trainable(self):
        return self.classifier.parameters()


class MobileNetV3Large_All(Model):
    def __init__(self, params: ParamsClassification):
        super().__init__(params)
        self.net = torchvision.models.mobilenet_v3_large(pretrained=True, progress=False).features
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(960, params.output_channels)

    def forward(self, x):
        x = self.net(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def trainable(self):
        return self.parameters()


class MobileNetV3Small_All(Model):
    def __init__(self, params: ParamsClassification):
        super().__init__(params)
        self.net = torchvision.models.mobilenet_v3_small(pretrained=True, progress=False).features

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(576, params.output_channels)

    def forward(self, x):
        x = self.net(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def trainable(self):
        return self.parameters()


class Learning(ABC):
    def __init__(self, params: ParamsClassification, model: Model,
                 optimizer_handle=torch.optim.Adam,
                 criterion_handle=nn.CrossEntropyLoss, draw_graph=False, string=None):

        self.params: ParamsClassification = params
        self.device = params.device
        self.str = model.__class__.__name__ + '_' + str(params) if string is None else string

        self.writer = SummaryWriter('runs/' + str(self)) if SummaryWriter is not None else None

        self.model = model.to(self.device)
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
            self.criterion = criterion_handle().to(self.device)
        else:
            self.criterion = None

        self.init_epoch = 0

        self.train_set = None
        self.valid_set = None
        self.test_set = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

    def __del__(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

    def __str__(self):
        return self.str

    def _load_train(self):
        self.train_set = FaceMaskSet(os.path.join(self.params.data_root, 'train'),
                                     self.params.transforms_train)
        self.train_loader = torch.utils.data.DataLoader(self.train_set,
                                                        batch_size=self.params.B, shuffle=True,
                                                        pin_memory=True, num_workers=num_workers)

    def _load_valid(self):
        self.valid_set = FaceMaskSet(os.path.join(self.params.data_root, 'valid'),
                                     self.params.transforms_test)
        self.valid_loader = torch.utils.data.DataLoader(self.valid_set,
                                                        batch_size=self.params.B, shuffle=False,
                                                        pin_memory=True, num_workers=num_workers)

    def _load_test(self):
        self.test_set = FaceMaskSet(os.path.join(self.params.data_root, 'test'),
                                    self.params.transforms_test)
        self.test_loader = torch.utils.data.DataLoader(self.test_set,
                                                       batch_size=1, shuffle=False,
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
                self.writer.add_scalar('Overall Accuracy/Train', accuracy_item, epoch)
            print('epoch: ', epoch, 'Training Loss: ', "%.5f" % loss_item,
                  'Overall Accuracy: ', "%.5f" % accuracy_item)

            self._validate(epoch)
            self.model.train()

            if epoch % checkpoint_interval == 0:
                self.save_model(epoch)

    def examine(self):
        self._load_valid()

        for i, batch in enumerate(self.valid_loader):
            bx = batch[0].to(self.device)
            prediction = self.model(bx)
            y_prime = torch.argmax(prediction, dim=1)
            self.plot_image(batch[0][0], y_prime.item(), batch[1].item())

    def export_to_onnx(self):
        dummy_tensor = torch.randn(size=(self.params.B,) + self.params.input_dims,
                                   device=self.params.device)
        torch.onnx.export(self.model, dummy_tensor, str(self) + '.onnx', verbose=True,
                          input_names=['x'], output_names=['output'],
                          dynamic_axes={'x': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}})

    def plot_image(self, img_tensor, prediction, ground_truth):
        fig, ax = plt.subplots(1)
        img = img_tensor.cpu().data

        # Display the image
        ax.imshow(img.permute(1, 2, 0))

        plt.title('Ground Truth: %s, Prediction: % s' % (
            self.train_set.classes[ground_truth], self.train_set.classes[prediction]))

        plt.show()

    def test(self):
        self._validate(self.init_epoch, mode='Test')

    def _validate(self, epoch, mode='Validation'):
        if mode == 'Validation':
            if self.valid_loader is None:
                self._load_valid()
            loader = self.valid_loader
            classes = self.valid_set.classes
        else:
            if self.test_loader is None:
                self._load_test()
            loader = self.test_loader
            classes = self.test_set.classes

        confusion_matrix = np.zeros((self.params.output_channels, self.params.output_channels),
                                    dtype=int)
        # row -> gt, col -> pred

        with torch.no_grad():
            total_time = 0

            self.model.eval()
            total_loss = torch.zeros(1, device=self.device)
            total_acc = torch.zeros(1, device=self.device)

            for i, batch in enumerate(tqdm(loader)):
                start_time = timer()
                bx = batch[0].to(self.device)
                by = batch[1].to(self.device)

                prediction = self.model(bx)
                time_added = (timer() - start_time) if 20 <= i < 120 else 0
                total_time += time_added
                loss = self.criterion(prediction, by)
                total_loss += loss
                y_prime = torch.argmax(prediction, dim=1)
                total_acc += torch.count_nonzero(torch.eq(y_prime, by))
                for g, p in zip(batch[1], y_prime):
                    confusion_matrix[g.item(), p.item()] += 1

            if i >= 120:
                total_time /= (100 * self.params.B)
            else:
                total_time /= ((i + 1) * self.params.B)
            total_time *= 1000  # [ms]

            loss_item = total_loss.item() / (i + 1)
            accuracy_item = total_acc.item() / (i + 1) / self.params.B

            class_acc = np.zeros(self.params.output_channels)
            for class_id in range(self.params.output_channels):
                class_acc[class_id] = confusion_matrix[class_id, class_id] / np.sum(
                        confusion_matrix[class_id])

            mean_acc = np.mean(class_acc)

            if self.writer is not None:
                self.writer.add_scalar('Loss/' + mode, loss_item, epoch)
                self.writer.add_scalar('Overall Accuracy/' + mode, accuracy_item, epoch)
                self.writer.add_scalar('Latency [ms]/' + mode, total_time, epoch)
                self.writer.add_scalar('Mean Accuracy/' + mode, mean_acc, epoch)
                for class_id in range(self.params.output_channels):
                    self.writer.add_scalar(
                            'Class Accuracy: ' + classes[class_id] + '/' + mode,
                            class_acc[class_id], epoch)
            print('epoch: ', epoch, mode + ' Loss: ', "%.5f" % loss_item,
                  'Overall Accuracy: ', "%.5f" % accuracy_item,
                  'Latency: ', "%.5f [ms]" % total_time,
                  'Mean Accuracy: ', '%.5f' % mean_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', help='Batch Size', default=32, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--gpu_id', help='GPU ID (0/1)', default=0, type=int)
    parser.add_argument('--model', default='MobileNetV3Large', help='Model Name')
    parser.add_argument('--epoch', default=-1, help='Load Epoch', type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--resize', default=-1, type=int)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--save', default=1, type=int, help='Checkpoint interval')
    parser.add_argument('--load', default='', help='Load Name')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--data_root', default='/home/zongyuez/data/Mask')
    parser.add_argument('--examine', action='store_true')
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)

    args = parser.parse_args()

    if not os.path.isdir('checkpoints/'):
        os.mkdir('checkpoints/')

    num_workers = args.num_workers
    device = 'cuda:' + str(args.gpu_id) if args.gpu_id >= 0 else 'cpu'

    params = ParamsClassification(B=args.batch, lr=args.lr, verbose=args.verbose,
                                  device=device, flip=args.flip,
                                  normalize=args.normalize,
                                  data_root=args.data_root,
                                  resize=args.resize)
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

    if args.examine:
        learner.examine()

    if args.export:
        learner.export_to_onnx()
