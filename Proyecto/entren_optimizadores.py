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
oculta_dim1 = 128
oculta_dim2 = 64
salida_dim = 10
tasa_aprendizaje = 0.01
num_epochs = 50 
batch_size = 64 

l2_lambda = 0.0001

capa1 = CapaDensa(entrada_dim, oculta_dim1, l2_lambda)
relu1 = ReLU()
capa2 = CapaDensa(oculta_dim1, oculta_dim2, l2_lambda)  # Segunda capa oculta
relu2 = ReLU()
capa3 = CapaDensa(oculta_dim2, salida_dim, l2_lambda)
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
    
    num_batches = total // batch_size  
    
    for batch in range(num_batches):
        inicio = batch * batch_size
        fin = inicio + batch_size

        x_batch = X_train[inicio:fin]  
        y_batch = Y_train[inicio:fin]  

        salida1 = capa1.forward(x_batch)
        salida_relu1 = relu1.forward(salida1)

        salida2 = capa2.forward(salida_relu1)
        salida_relu2 = relu2.forward(salida2)

        salida3 = capa3.forward(salida_relu2)
        salida_softmax = softmax.forward(salida3)

        loss = loss_fn.forward(salida_softmax, y_batch, [capa1, capa2])
        loss_total += loss

        predicciones = np.argmax(salida_softmax, axis=1)
        etiquetas_verdaderas = np.argmax(y_batch, axis=1)
        correctos += np.sum(predicciones == etiquetas_verdaderas)

        grad_loss = loss_fn.backward()  
        grad_softmax = capa3.backward(grad_loss, tasa_aprendizaje)
        grad_relu2 = relu2.backward(grad_softmax)
        grad_capa2 = capa2.backward(grad_relu2, tasa_aprendizaje)
        grad_relu1 = relu1.backward(grad_capa2)
        capa1.backward(grad_relu1, tasa_aprendizaje)

    optimizer.post_update_params()  

    precision = correctos / (num_batches * batch_size) * 100
    pérdidas.append(loss_total / num_batches)
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

for i in range(20): 
    x = X_train[i:i+1]  
    y_true = Y_train[i:i+1]  

    salida1 = capa1.forward(x)
    salida_relu1 = relu1.forward(salida1)

    salida2 = capa2.forward(salida_relu1)
    salida_relu2 = relu2.forward(salida2)

    salida3 = capa3.forward(salida_relu2)
    salida_softmax = softmax.forward(salida3)
    
    prediccion = np.argmax(salida_softmax)
    etiqueta_verdadera = np.argmax(y_true)

    plt.imshow(x.reshape(28, 28), cmap="gray")
    plt.title(f"Predicción: {prediccion}, Verdadera: {etiqueta_verdadera}")
    plt.show()