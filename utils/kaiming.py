import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")

        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)

        if m.bias is not None:
            nn.init.zeros_(m.bias)
