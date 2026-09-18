from torch.utils.data import DataLoader

from .dataset import get_datasets


def get_loaders(batch_size=128):
    train_set, val_set, test_set = get_datasets()

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader
