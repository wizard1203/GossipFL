import logging

import torch
import torchvision.models as models

from model.cv.resnet_gn import resnet18
from model.cv.resnet_v2 import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNet10
from model.cv.resnet_b import resnet20
from model.cv.mnistflnet import MnistFLNet
from model.cv.cifar10flnet import Cifar10FLNet

from data_preprocessing.utils.stats import get_dataset_image_size




CV_MODEL_LIST = []
RNN_MODEL_LIST = ["rnn"]


def create_model(args, model_name, output_dim, pretrained=False, **kwargs):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    model = None
    logging.info(f"model name: {model_name}")

    if model_name in RNN_MODEL_LIST:
        pass
    else:
        image_size = get_dataset_image_size(args.dataset)

    if model_name == "mnistflnet" and args.dataset in ["mnist", "fmnist"]:
        logging.info("MnistFLNet + MNIST or FMNIST")
        model = MnistFLNet(input_channels=args.model_input_channels, output_dim=output_dim)

    elif model_name == "cifar10flnet" and args.dataset == "cifar10":
        logging.info("Cifar10FLNet + CIFAR-10")
        model = Cifar10FLNet()
    elif model_name == "resnet18_gn" or model_name == "resnet18":
        logging.info("ResNet18_GN or resnet18")
        model = resnet18(pretrained=pretrained, num_classes=output_dim, group_norm=args.group_norm_num)
    elif model_name == "resnet18_v2":
        logging.info("ResNet18_v2")
        model = ResNet18(args=args, num_classes=output_dim, image_size=image_size,
                        model_input_channels=args.model_input_channels)
    elif model_name == "resnet34_v2":
        logging.info("ResNet34_v2")
        model = ResNet34(args=args, num_classes=output_dim, image_size=image_size,
                        model_input_channels=args.model_input_channels)
    elif model_name == "resnet10_v2":
        logging.info("ResNet10_v2")
        model = ResNet10(args=args, num_classes=output_dim, image_size=image_size,
                        model_input_channels=args.model_input_channels)
    elif model_name == "resnet20":
        model = resnet20(num_classes=output_dim)
    else:
        logging.info(f"model name is {model_name}")
        raise NotImplementedError

    return model


