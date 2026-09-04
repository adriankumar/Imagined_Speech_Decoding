#contains arbitrary components/modules to apply to model
#i.e attention head, convolution head, lora adapter

from .convolution_head_2d import ConvHead_2D
from .multi_attention_head import MAH
from .adapters import LoRA
from .quantile_head import QuantileHead
from .MLP import MLPClassifier