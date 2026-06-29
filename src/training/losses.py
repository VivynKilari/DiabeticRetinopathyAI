import torch.nn as nn


def get_loss_function():
    """
    Standard CrossEntropyLoss.
    Later we'll replace this with Focal Loss.
    """
    return nn.CrossEntropyLoss()