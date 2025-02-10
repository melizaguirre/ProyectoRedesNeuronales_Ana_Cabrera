import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def encode_to_numbers(word):
    word = word.lower() 
    if len(word) > 8:
        raise ValueError("La palabra no debe de contener más de 8 caracteres")
    if not word.isalpha():  
        raise ValueError("La palabra debe contener solo letras del alfabeto")
    start_on1 = [ord(c) - ord('a') + 1 for c in word]
    
    while len(start_on1) < 8:
        start_on1.append(0)
    
    return np.array(start_on1)

palindromes = ["radar", "level", "civic", "rotor", "kayak", "madam", "refer", "racecar", "deed", "noon", "pop", "solos", "stats", "wow", "tenet"]
no_palindromes = ["apple", "banana", "orange", "pencil", "tiger", "monkey", "school", "guitar", "bottle", "travel", "rocket", "jumper", "dragon", "laptop", "engine"]

data = palindromes + no_palindromes
labels = [1] * len(palindromes) + [0] * len(no_palindromes)

X = np.array([encode_to_numbers(word) for word in data])
y = np.array(labels)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)


def is_palindrome(word_vector):
    entrada = 8   
    capa_oculta_1 = 16  
    capa_oculta_2 = 8  
    salida = 1  

    np.random.seed(42)  

    W1 = np.random.randn(entrada, capa_oculta_1) * 0.01  
    b1 = np.random.randn(capa_oculta_1) * 0.01  

    W2 = np.random.randn(capa_oculta_1, capa_oculta_2) * 0.01  
    b2 = np.random.randn(capa_oculta_2) * 0.01  

    W3 = np.random.randn(capa_oculta_2, salida) * 0.01 
    b3 = np.random.randn(salida) * 0.01  

    entrada_capa_oculta1 = np.dot(word_vector, W1) + b1
    salida_capa_oculta1 = relu(entrada_capa_oculta1)

    entrada_capa_oculta2 = np.dot(salida_capa_oculta1, W2) + b2
    salida_capa_oculta2 = relu(entrada_capa_oculta2)

    output_layer_input = np.dot(salida_capa_oculta2, W3) + b3
    predicted_output = sigmoid(output_layer_input)

    return predicted_output.item()


def predict_palindrome():
    word = input("Introduce una palabra para analizar (máximo 8 caracteres): ").lower()

    try:
    
        word_vector = encode_to_numbers(word)
        print(f"Palabra codificada: {word_vector}")
        
        output = is_palindrome(word_vector)
        
        if output > 0.5:
            print("La palabra es probablemente un palíndromo")
        else:
            print("La palabra NO es un palíndromo")
        
    except ValueError as e:
        print(e)

predict_palindrome()
