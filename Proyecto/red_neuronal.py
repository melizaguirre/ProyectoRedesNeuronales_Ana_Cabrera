import numpy as np

class CapaDensa:
    def __init__(self, entradas: int, neuronas: int, l2_lambda: float) -> None:
        self.pesos=np.random.rand(entradas, neuronas) *0.01
        self.sesgos=np.zeros((1, neuronas))
        self.l2_lambda = l2_lambda

    def forward(self, datos_entrada):
        self.entrada = datos_entrada
        self.salida = np.dot(datos_entrada, self.pesos)+ self.sesgos
        return self.salida  

    def backward(self, grad_salida, tasa_aprendizaje=0.1):
        grad_pesos = np.dot(self.entrada.T, grad_salida) + self.l2_lambda * self.pesos
        grad_sesgos = np.sum(grad_salida, axis=0, keepdims=True)
        grad_entrada = np.dot(grad_salida, self.pesos.T)
        
        self.pesos -= tasa_aprendizaje * grad_pesos
        self.sesgos -= tasa_aprendizaje * grad_sesgos
        
        return grad_entrada

class ReLU:
    def forward(self, x):
        self.entrada = x
        return np.maximum(0, x)

    def backward(self, grad_salida):
        grad_entrada = grad_salida * (self.entrada > 0)
        return grad_entrada

class Softmax:
    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis = 1, keepdims= True))
        self.salida = exp_x / np.sum(exp_x, axis = 1, keepdims=True)
        return self.salida
        
        def backward(self, grad_salida):
            batch_size = grad_salida.shape[0]
        grad_entrada = np.zeros_like(grad_salida)
        
        for i in range(batch_size):
            softmax_output = self.salida[i]
            
            jacobian = np.diag(softmax_output) - np.outer(softmax_output, softmax_output)
            grad_entrada[i] = np.dot(jacobian, grad_salida[i])
        return grad_salida
