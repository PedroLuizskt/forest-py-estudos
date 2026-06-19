"""Redes neurais convolucionais (CNN) em PyTorch para tarefas geoespaciais."""

from forestpy.ml.cnn.architectures import SimpleCNN
from forestpy.ml.cnn.segmentation_trainer import UNetTrainer
from forestpy.ml.cnn.trainer import CNNTrainer
from forestpy.ml.cnn.unet import UNet

__all__ = ["SimpleCNN", "CNNTrainer", "UNet", "UNetTrainer"]
