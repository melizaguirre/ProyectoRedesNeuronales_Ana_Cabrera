import numpy as np 
from red_neuronal import CapaDensa, ReLU, Softmax
from funciones_perdida import crossEntropyLoss
from Mnist.MnistDataset import MnistDataset

mnist = MnistDataset()
mnist.load("Mnist/dataset/train-images-idx3-ubyte", "Mnist/dataset/train-labels-idx1-ubyte")
X_train = mnist.images
Y_train = mnist.one_hot_labels

entrada_dim = 784
oculta_dim = 128
salida_dim = 10
tasa_aprendizaje = 0.1
num_epochs = 10  


capa1 = CapaDensa(entrada_dim, oculta_dim)
relu = ReLU()
capa2 = CapaDensa(oculta_dim, salida_dim)
softmax = Softmax()
loss_fn = crossEntropyLoss()

for epoch in range(num_epochs):
    loss_total = 0
    for i in range(len(X_train)):
        x = X_train[i:i+1]  
        y = Y_train[i:i+1]  

        salida1 = capa1.forward(x)
        salida_relu = relu.forward(salida1)
        salida2 = capa2.forward(salida_relu)
        salida_softmax = softmax.forward(salida2)

        loss = loss_fn.forward(salida_softmax, y)
        loss_total += loss

        grad_loss = loss_fn.backward()
        grad_softmax = capa2.backward(grad_loss, tasa_aprendizaje)
        grad_relu = relu.backward(grad_softmax)
        capa1.backward(grad_relu, tasa_aprendizaje)

    print(f"Época {epoch+1}/{num_epochs}, Pérdida: {loss_total/len(X_train)}")

print("Entrenamiento finalizado.")