import numpy as np
import Redes_Neuronales as rn


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

palindromes = ["oso", "radar", "level", "civic", "rotor", "kayak", "madam", "refer", "racecar", "deed", "noon", "pop", "solos", "stats", "wow", "tenet"]
no_palindromes = ["apple", "banana", "orange", "pencil", "tiger", "monkey", "school", "guitar", "bottle", "travel", "rocket", "jumper", "dragon", "laptop", "engine"]

data = palindromes + no_palindromes
labels = [1] * len(palindromes) + [0] * len(no_palindromes)

X = np.array([encode_to_numbers(word) for word in data])
y = np.array(labels)

def is_palindrome(word_vector):
    capa1 = rn.capaDensa(8, 200)
    capa1.forward(word_vector)
    relu1 = rn.ReLU()
    relu1.forward(capa1.salida)
    
    capa2 = rn.capaDensa(200, 80)
    capa2.forward(relu1.salida)
    relu2 = rn.ReLU()
    relu2.forward(capa2.salida)
    
    capaSalida = rn.capaDensa(80, 1)
    capaSalida.forward(relu2.salida)
    sigmoide_salida = rn.Sigmoide()
    sigmoide_salida.forward(capaSalida.salida)
    
    return sigmoide_salida.salida.item()

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
