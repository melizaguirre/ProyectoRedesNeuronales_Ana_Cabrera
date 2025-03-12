import numpy as np 
from red_neuronal import CapaDensa, ReLU, Softmax
from funciones_perdida import crossEntropyLoss
from MnistLoader import MnistDataset

entrada_dim = 784
oculta_dim = 128
salida_dim = 10
tasa_aprendizaje = 0.1