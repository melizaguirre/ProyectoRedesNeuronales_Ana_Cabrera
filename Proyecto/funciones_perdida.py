import numpy as np
class crossEntropyLoss:
    def forward(self, y_pred, y_true):
        batch_size = y_pred.shape[0]

        self.y_pred = y_pred  
        self.y_true = y_true
        epsilon = 1e-15
        self.y_pred = np.clip(self.y_pred, epsilon, 1 - epsilon)  

        loss = -np.sum(self.y_true * np.log(self.y_pred)) / batch_size
        return loss
    def backward (self):
        return self.y_pred - self.y_true