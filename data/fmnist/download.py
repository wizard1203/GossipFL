from torchvision.datasets import FashionMNIST

data_dir = '.'
mnist_dataobj = FashionMNIST(data_dir, train=True, download=True)
mnist_dataobj = FashionMNIST(data_dir, train=False, download=True)


