import os
from torchvision.datasets import CIFAR10, CIFAR100
from xautodl.datasets.DownsampledImageNet import ImageNet16
import torchvision.transforms as t


def cifar10_builder(root, size=None):
    size = size or 32
    root = os.path.join(root, 'CIFAR10')

    print(f"Input size: {size}x{size}")

    d_train = CIFAR10(root=root, train=True, download=True, transform=t.Compose([
        t.RandomCrop(size, padding=4),
        t.RandomHorizontalFlip(),
        t.ToTensor(),
        t.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]))

    d_test = CIFAR10(root=root, train=False, download=True, transform=t.Compose([
        *((t.Resize(size),) if size != 32 else ()),
        t.ToTensor(),
        t.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]))

    return d_train, d_test


def cifar100_builder(root, size=None):
    size = size or 32
    root = os.path.join(root, 'CIFAR100')

    print(f"Input size: {size}x{size}")

    d_train = CIFAR100(root=root, train=True, download=True, transform=t.Compose([
        t.RandomCrop(size, padding=4),
        t.RandomHorizontalFlip(),
        t.ToTensor(),
        t.Normalize((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
    ]))

    d_test = CIFAR100(root=root, train=False, download=True, transform=t.Compose([
        *((t.Resize(size),) if size != 32 else ()),
        t.ToTensor(),
        t.Normalize((0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
    ]))

    return d_train, d_test


def imagenet16_builder(root, size=None):
    size = size or 16
    # "size" argument is here just for compatibility
    assert size == 16

    root = os.path.join(root, 'ImageNet16')

    print(f"Input size: {size}x{size}")

    d_train = ImageNet16(root=root, train=True, transform=t.Compose([
        t.RandomCrop(size, padding=2),
        t.RandomHorizontalFlip(),
        t.ToTensor(),
        t.Normalize((0.4810980392156863, 0.45749019607843133, 0.4078823529411765),
                    (0.247921568627451, 0.24023529411764705, 0.2552549019607843)),
    ]), use_num_of_class_only=120)

    d_test = ImageNet16(root=root, train=False, transform=t.Compose([
        # *((t.Resize(size),) if size != 32 else ()),
        t.ToTensor(),
        t.Normalize((0.4810980392156863, 0.45749019607843133, 0.4078823529411765),
                    (0.247921568627451, 0.24023529411764705, 0.2552549019607843)),
    ]), use_num_of_class_only=120)

    return d_train, d_test


DATASET_BUILDERS = {
    'cifar10': cifar10_builder,
    'cifar100': cifar100_builder,
    'ImageNet16-120': imagenet16_builder
}
