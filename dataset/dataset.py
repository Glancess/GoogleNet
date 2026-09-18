from torchvision import datasets, transforms
from torch.utils.data import Subset


def get_datasets(root="./data"):

    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    val_test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    # 同一份官方 50000 train 数据
    # 但建立两个对象，因为以后 train / val transform 会不同
    train_dataset = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=train_transform,
    )

    val_dataset = datasets.CIFAR10(
        root=root,
        train=True,
        download=False,
        transform=val_test_transform,
    )

    # 按论文思路：
    # 0 ~ 39999 -> train
    # 40000 ~ 49999 -> val
    train_indices = range(0, 40000)
    val_indices = range(40000, 50000)

    train_set = Subset(train_dataset, train_indices)
    val_set = Subset(val_dataset, val_indices)

    # 官方 test
    test_set = datasets.CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=val_test_transform,
    )

    return train_set, val_set, test_set
