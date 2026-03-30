"""
Aula 06: Aceleração com Hardware: GPU, TPU e Aceleradores

Pacote contendo módulos para demonstração de técnicas de aceleração
de hardware em Machine Learning, incluindo Mixed Precision Training,
Gradient Checkpointing e profiling de GPU.

Referências:
    Micikevicius, P. et al. (2018). Mixed Precision Training. ICLR.
    Chen, T. et al. (2016). Training Deep Nets with Sublinear Memory Cost.
    Williams, S. et al. (2009). Roofline: An Insightful Visual Performance Model.
"""

from src.data_preprocessing import DataPreprocessor
from src.model import SimpleViT, create_model
from src.training import Trainer, train_model
from src.evaluation import ModelEvaluator, calculate_metrics
from src.utils import get_device, save_model, load_model, save_metrics

__all__ = [
    "DataPreprocessor",
    "SimpleViT",
    "create_model",
    "Trainer",
    "train_model",
    "ModelEvaluator",
    "calculate_metrics",
    "get_device",
    "save_model",
    "load_model",
    "save_metrics",
]
