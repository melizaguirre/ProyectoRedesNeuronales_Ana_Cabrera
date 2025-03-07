import numpy as np

class CapaDensa:
    def __init__(self, entradas: int, neuronas: int) -> None:
        self.pesos=np.random.rand(entradas, neuronas) *0.01
        self.sesgos=np.zeros((1, neuronas))

    def forward(self, datos_entrada):
        self.entrada = datos_entrada
        self.salida = np.dot(datos_entrada, self.pesos)+ self.sesgos
        return self.salida  