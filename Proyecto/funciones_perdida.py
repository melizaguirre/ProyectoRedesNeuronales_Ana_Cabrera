import numpy as np
class crossEntropyLoss:
    def forward(self, y_pred, y_true):
        batch_size = y_pred.shape[0]

        epsilon = 1e-15
        y_pred= np.clip(y_pred, epsilon, 1-epsilon)

        loss = -np.sum(y_true * np.log(y_pred)) / batch_size
        return loss
    def backward (self, y_pred, y_true):
        batch_size = y_pred.shape[0]
        return self.y_pred - self.y_true