import numpy as np
class crossEntropyLoss:
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    """def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        m = y_true.shape[0]  # tamaño del batch
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)  
        log_probs = -np.log(y_pred)
        loss = np.sum(log_probs * y_true) / m
        return loss"""
        
    def forward(self, y_pred, y_true, capas):
        batch_size = y_pred.shape[0]
        self.y_pred = y_pred
        self.y_true = y_true

        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        loss = -np.sum(y_true * np.log(y_pred)) / batch_size

        l2_loss = sum(np.sum(capa.pesos ** 2) for capa in capas if hasattr(capa, 'pesos'))
        loss += (0.0001 / 2) * l2_loss  

        return loss

    def backward(self):
        grad = self.y_pred - self.y_true
        return grad
