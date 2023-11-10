import logging

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import CIFAR100

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar100():
    CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
    CIFAR_STD = [0.2673, 0.2564, 0.2762]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform


def load_centralized_cifar100(dataset, data_dir, batch_size, 
                  max_train_len=None, max_test_len=None,
                  args=None):

    train_transform, test_transform = _data_transforms_cifar100()

    train_dataset = CIFAR100(root=data_dir, train=True,
                            transform=train_transform, download=False)

    test_dataset = CIFAR100(root=data_dir, train=False,
                            transform=test_transform, download=False)

    if max_train_len is not None:
        train_dataset.data = train_dataset.data[0: max_train_len]
        train_dataset.target = np.array(train_dataset.targets)[0: max_train_len]
    shuffle = True

    train_dl = data.DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=shuffle, num_workers=4)
    test_dl = data.DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=4)

    class_num = 100

    train_data_num = len(train_dataset)
    test_data_num = len(test_dataset)

    return train_dl, test_dl, train_data_num, test_data_num, class_num




