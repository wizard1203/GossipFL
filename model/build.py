import logging

from model.cv.cnn import CNN_DropOut
from model.cv.simplecnn_mnist import SimpleCNNMNIST 
from model.cv.resnet_gn import resnet18
from model.cv.resnet_b import resnet20
from model.cv.mobilenet import mobilenet
from model.cv.resnet import resnet56
from model.cv.mobilenet_v3 import MobileNetV3
from model.cv.efficientnet import EfficientNet
from model.cv.mnistflnet import MnistFLNet
from model.cv.cifar10flnet import Cifar10FLNet

from model.cv.others import (ModerateCNNMNIST, ModerateCNN)



def create_model(args, model_name, output_dim, **kargs):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    model = None
    if model_name == "simplecnn_mnist" and args.dataset in ["mnist", "fmnist"]:
        logging.info("simplecnn_mnist + MNIST or FMNIST")
        model = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
    elif model_name == "mnistflnet" and args.dataset in ["mnist", "fmnist"]:
        logging.info("MnistFLNet + MNIST or FMNIST")
        model = MnistFLNet()
    elif model_name == "cifar10flnet" and args.dataset == "cifar10":
        logging.info("Cifar10FLNet + CIFAR-10")
        model = Cifar10FLNet()
    elif model_name == "cnn" and args.dataset == "femnist":
        logging.info("CNN + FederatedEMNIST")
        model = CNN_DropOut(False)
    elif model_name == "vgg-9":
        if args.dataset in ("mnist", 'femnist'):
            model = ModerateCNNMNIST()
        elif args.dataset in ("cifar10", "cifar100", "cinic10", "svhn"):
            # print("in moderate cnn")
            model = ModerateCNN()
        elif args.dataset == 'celeba':
            model = ModerateCNN(output_dim=2)
    elif model_name == "resnet18_gn":
        logging.info("ResNet18_GN")
        model = resnet18(pretrained=args.pretrained, num_classes=output_dim, group_norm=args.group_norm_num)
    elif model_name == "resnet20":
        model = resnet20(num_classes=output_dim)
    elif model_name == "resnet56":
        model = resnet56(class_num=output_dim)
    elif model_name == "mobilenet":
        model = mobilenet(class_num=output_dim)
    # TODO
    elif model_name == 'mobilenet_v3':
        '''model_mode \in {LARGE: 5.15M, SMALL: 2.94M}'''
        model = MobileNetV3(model_mode='LARGE', num_classes=output_dim)
    elif model_name == 'efficientnet':
        # model = EfficientNet()
        efficientnet_dict = {
            # Coefficients:   width,depth,res,dropout
            'efficientnet-b0': (1.0, 1.0, 224, 0.2),
            'efficientnet-b1': (1.0, 1.1, 240, 0.2),
            'efficientnet-b2': (1.1, 1.2, 260, 0.3),
            'efficientnet-b3': (1.2, 1.4, 300, 0.3),
            'efficientnet-b4': (1.4, 1.8, 380, 0.4),
            'efficientnet-b5': (1.6, 2.2, 456, 0.4),
            'efficientnet-b6': (1.8, 2.6, 528, 0.5),
            'efficientnet-b7': (2.0, 3.1, 600, 0.5),
            'efficientnet-b8': (2.2, 3.6, 672, 0.5),
            'efficientnet-l2': (4.3, 5.3, 800, 0.5),
        }
        # default is 'efficientnet-b0'
        model = EfficientNet.from_name(
            model_name='efficientnet-b0', num_classes=output_dim)
        # model = EfficientNet.from_pretrained(model_name='efficientnet-b0')
    else:
        raise NotImplementedError
    return model


