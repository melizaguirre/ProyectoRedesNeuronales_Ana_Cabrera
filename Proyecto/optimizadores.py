import numpy as np
class Optimizer_SGD:
    def __init__(self, learning_rate=1.0, decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.iterations = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer, grad_pesos, grad_sesgos):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.pesos)
                layer.bias_momentums = np.zeros_like(layer.sesgos)

            weight_updates = self.momentum * layer.weight_momentums - \
                             self.current_learning_rate * grad_pesos
            bias_updates = self.momentum * layer.bias_momentums - \
                           self.current_learning_rate * grad_sesgos

            layer.weight_momentums = weight_updates
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * grad_pesos
            bias_updates = -self.current_learning_rate * grad_sesgos

        layer.pesos += weight_updates
        layer.sesgos += bias_updates

    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta1=0.9, beta2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer, grad_pesos, grad_sesgos):
        assert grad_pesos.shape == layer.pesos.shape, f"Error: grad_pesos.shape {grad_pesos.shape} != layer.pesos.shape {layer.pesos.shape}"
        assert grad_sesgos.shape == layer.sesgos.shape, f"Error: grad_sesgos.shape {grad_sesgos.shape} != layer.sesgos.shape {layer.sesgos.shape}"

        if not hasattr(layer, 'weight_momentums') or layer.weight_momentums.shape != layer.pesos.shape:
            layer.weight_momentums = np.zeros_like(layer.pesos)
            layer.bias_momentums = np.zeros_like(layer.sesgos)
            layer.weight_cache = np.zeros_like(layer.pesos)
            layer.bias_cache = np.zeros_like(layer.sesgos)

        layer.weight_momentums = self.beta1 * layer.weight_momentums + (1 - self.beta1) * grad_pesos
        layer.bias_momentums = self.beta1 * layer.bias_momentums + (1 - self.beta1) * grad_sesgos

        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta1 ** (self.iterations + 1))

        layer.weight_cache = self.beta2 * layer.weight_cache + (1 - self.beta2) * grad_pesos**2
        layer.bias_cache = self.beta2 * layer.bias_cache + (1 - self.beta2) * grad_sesgos**2

        weight_cache_corrected = layer.weight_cache / (1 - self.beta2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta2 ** (self.iterations + 1))

        layer.pesos -= self.current_learning_rate * weight_momentums_corrected / \
                       (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.sesgos -= self.current_learning_rate * bias_momentums_corrected / \
                        (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1
