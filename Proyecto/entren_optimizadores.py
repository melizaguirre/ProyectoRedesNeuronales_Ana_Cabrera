import numpy as np 
import matplotlib.pyplot as plt
from red_neuronal import CapaDensa, ReLU, Softmax
from funciones_perdida import crossEntropyLoss
from Mnist.MnistDataset import MnistDataset
from optimizadores import Optimizer_SGD, Optimizer_Adam


mnist = MnistDataset()
mnist.load("Mnist/dataset/train-images-idx3-ubyte", "Mnist/dataset/train-labels-idx1-ubyte")
X_train = mnist.images
Y_train = mnist.one_hot_labels


entrada_dim = 784
oculta_dim = 128
salida_dim = 10
tasa_aprendizaje = 0.01
num_epochs = 50  

l2_lambda = 0.001

capa1 = CapaDensa(entrada_dim, oculta_dim, l2_lambda)
relu = ReLU()
capa2 = CapaDensa(oculta_dim, salida_dim, l2_lambda)
softmax = Softmax()
loss_fn = crossEntropyLoss()


optimizer = Optimizer_Adam(learning_rate=0.001, decay=1e-4)
# optimizer = Optimizer_SGD(learning_rate=0.01, decay=1e-4, momentum=0.9)

pérdidas = []
precisiones = []

for epoch in range(num_epochs):
    optimizer.pre_update_params()
    loss_total = 0
    correctos = 0  
    total = len(X_train)

    for i in range(total):
        x = X_train[i:i+1]  
        y = Y_train[i:i+1]  

        
        salida1 = capa1.forward(x)
        salida_relu = relu.forward(salida1)
        salida2 = capa2.forward(salida_relu)
        salida_softmax = softmax.forward(salida2)

        loss = loss_fn.forward(salida_softmax, y, [capa1, capa2])
        loss_total += loss

        
        prediccion = np.argmax(salida_softmax)
        etiqueta_verdadera = np.argmax(y)
        if prediccion == etiqueta_verdadera:
            correctos += 1

        
        grad_loss = loss_fn.backward()  
        grad_softmax = capa2.backward(grad_loss, tasa_aprendizaje)  
        grad_relu = relu.backward(grad_softmax)
        capa1.backward(grad_relu)

    optimizer.post_update_params()  

    precision = correctos / total * 100

    pérdidas.append(loss_total / total)
    precisiones.append(precision)

    print(f"Época {epoch+1}/{num_epochs}, Pérdida: {pérdidas[-1]:.4f}, Precisión: {precisiones[-1]:.2f}%")

precision_total = np.mean(precisiones)
print(f"\n Precisión total promedio en {num_epochs} épocas: {precision_total:.2f}%")

plt.plot(range(1, num_epochs+1), pérdidas, label="Pérdida")
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.title('Pérdida durante el entrenamiento')
plt.legend()
plt.show()

plt.plot(range(1, num_epochs+1), precisiones, label="Precisión", color="green")
plt.xlabel('Épocas')
plt.ylabel('Precisión (%)')
plt.title('Precisión durante el entrenamiento')
plt.legend()
plt.show()

print("Entrenamiento finalizado.")

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