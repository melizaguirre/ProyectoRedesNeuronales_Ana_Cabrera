import numpy as np 
import matplotlib.pyplot as plt
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
num_epochs = 40  


capa1 = CapaDensa(entrada_dim, oculta_dim)
relu = ReLU()
capa2 = CapaDensa(oculta_dim, salida_dim)
softmax = Softmax()
loss_fn = crossEntropyLoss()

pérdidas = []

for epoch in range(num_epochs):
    loss_total = 0
    correctos_epoch = 0

    for i in range(len(X_train)):
        x = X_train[i:i+1]  
        y = Y_train[i:i+1]  

        salida1 = capa1.forward(x)
        salida_relu = relu.forward(salida1)
        salida2 = capa2.forward(salida_relu)
        salida_softmax = softmax.forward(salida2)

        loss = loss_fn.forward(salida_softmax, y)
        loss_total += loss

        prediccion = np.argmax(salida_softmax)
        etiqueta_verdadera = np.argmax(y)
        if prediccion == etiqueta_verdadera:
            correctos_epoch += 1 

        grad_loss = loss_fn.backward()
        grad_softmax = capa2.backward(grad_loss, tasa_aprendizaje)
        grad_relu = relu.backward(grad_softmax)
        capa1.backward(grad_relu, tasa_aprendizaje)
    precision_epoch = (correctos_epoch / len(X_train)) * 100

    print(f"Época {epoch+1}/{num_epochs}, Pérdida: {loss_total/len(X_train):.4f}, Precisión: {precision_epoch:.2f}%")
    pérdidas.append(loss_total / len(X_train))


plt.plot(range(num_epochs), pérdidas)
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.title('Pérdida durante el entrenamiento')
plt.show()

for i in range(10): 
    x = X_train[i:i+1]  
    y_true = Y_train[i:i+1]  
    salida1 = capa1.forward(x)
    salida_relu = relu.forward(salida1)
    salida2 = capa2.forward(salida_relu)
    salida_softmax = softmax.forward(salida2)
    
    prediccion = np.argmax(salida_softmax)
    etiqueta_verdadera = np.argmax(y_true)

    plt.imshow(x.reshape(28, 28), cmap="gray")
    plt.title(f"Predicción: {prediccion}, Verdadera: {etiqueta_verdadera}")
    plt.show()

correctos = 0
total = len(X_train)

for i in range(total):
    x = X_train[i:i+1]  
    y_true = Y_train[i:i+1]  
    salida1 = capa1.forward(x)
    salida_relu = relu.forward(salida1)
    salida2 = capa2.forward(salida_relu)
    salida_softmax = softmax.forward(salida2)

    prediccion = np.argmax(salida_softmax)
    etiqueta_verdadera = np.argmax(y_true)

    if prediccion == etiqueta_verdadera:
        correctos += 1

precision = correctos / total * 100
print(f"Precisión: {precision}%")
print("Entrenamiento finalizado.")