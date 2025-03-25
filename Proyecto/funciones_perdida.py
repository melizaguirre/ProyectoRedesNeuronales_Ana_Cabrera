import numpy as np
class crossEntropyLoss:
    def __init__(self, l2_lambda=0.0001):
        self.y_pred = None
        self.y_true = None
        self.l2_lambda = l2_lambda 

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
        self.y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15) 
        self.y_true = y_true

        loss = -np.sum(y_true * np.log(self.y_pred)) / batch_size

        if self.l2_lambda > 0:
            l2_loss = sum(np.sum(capa.pesos ** 2) for capa in capas if hasattr(capa, 'pesos'))
            loss += (self.l2_lambda / 2) * l2_loss  

        return loss

    def backward(self):
        return (self.y_pred - self.y_true) / self.y_true.shape[0] 
