import numpy as np

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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)


def is_palindrome(word_vector):
    entrada = 8   
    capa_oculta = 16  
    salida = 1  

    np.random.seed(42)  

    W1 = np.random.randn(entrada, capa_oculta)  
    b1 = np.random.randn(capa_oculta)  

    W2 = np.random.randn(capa_oculta, salida) 
    b2 = np.random.randn(salida) 

    entrada_capa_oculta = np.dot(word_vector, W1) + b1
    salida_Capa_oculta = relu(entrada_capa_oculta)  
    
    output_layer_input = np.dot(salida_Capa_oculta, W2) + b2
    predicted_output = sigmoid(output_layer_input)  
    
    return predicted_output


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
