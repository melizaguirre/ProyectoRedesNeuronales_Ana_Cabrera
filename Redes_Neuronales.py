import numpy as np

class capaDensa:
    def __init__(self, entradas: int, neuronas: int):
        self.pesos = np.random.randn(entradas,neuronas) * 0.01
        self.sesgos = np.zeros((1, neuronas))

    def forward(self, datos:list[float]):
        self.salida = np.dot(datos,self.pesos) + self.sesgos

class ReLU:
    def forward(self,x:list[float]) -> None:
            self.salida= np.maximum(0,x)

class Softmax:
    def forward(self,x:list[float]):
            exp_x = np.exp(x-np.max(x))
            self.salida = exp_x / np.sum(exp_x,axis=1)

class Sigmoide:
    def forward(self,x:list[float]):
        self.salida = 1/(1+np.exp(-x))
