import numpy as np
import pickle
import matplotlib.pyplot as plt
from red_neuronal import CapaDensa, ReLU, Softmax
from Mnist.MnistDataset import MnistDataset


mnist = MnistDataset()
mnist.load("Mnist/dataset/train-images-idx3-ubyte", "Mnist/dataset/train-labels-idx1-ubyte")
#mnist.load("Mnist/dataset/t10k-images-idx3-ubyte", "Mnist/dataset/t10k-labels-idx1-ubyte")
X_test = mnist.images[:20]  
Y_test = mnist.one_hot_labels[:20]


entrada_dim = 784
oculta_dim1 = 128
oculta_dim2 = 128
salida_dim = 10
l2_lambda = 0.0005


capa1 = CapaDensa(entrada_dim, oculta_dim1, l2_lambda)
relu1 = ReLU()
capa2 = CapaDensa(oculta_dim1, oculta_dim2, l2_lambda)
relu2 = ReLU()
capa3 = CapaDensa(oculta_dim2, salida_dim, l2_lambda)
softmax = Softmax()


with open("pesos_red.pkl", "rb") as archivo:
    pesos = pickle.load(archivo)

capa1.pesos = pesos["capa1_pesos"]
capa1.bias = pesos["capa1_bias"]
capa2.pesos = pesos["capa2_pesos"]
capa2.bias = pesos["capa2_bias"]
capa3.pesos = pesos["capa3_pesos"]
capa3.bias = pesos["capa3_bias"]

print("Pesos cargados correctamente. Realizando predicciones...")


for i in range(20): 
    x = X_test[i:i+1]  
    y_true = Y_test[i:i+1]  

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
