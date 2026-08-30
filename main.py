"""
    Juan Pablo Arteaga Patiño | A01665795
    Primer Implementación Módulo 2 Algoritmo de Clasificación Manual
"""
from pathlib import Path


# imports
import pandas as pd
import numpy as np
import matplotlib as plt

# load dataset
def load_data():
    ROOT = Path(__file__).resolve().parent
    data = pd.read_csv(ROOT / "Dry_Bean_Dataset.csv")
    return data

# Load data
bean = load_data()
print(bean.head())

"""
    El dataset no requiere de mucho preprocesamiento, ya que no tiene valores nulos y todos los features son numéricos.
    Tampoco contiene outliers.
    ___ Mas info en el readme ___
"""

# funcion de activacion softmax
def g(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# funcion de costo usando one-hot encoding
def cross_ent(y, y_hat):
    # aqui "y" es un vector de 0s y 1s, y "y_hat" es un vector en el rango (0,1)
    return -np.sum(y * np.log(y_hat))

# funcion de predicciones para un batch
def pred_func(X, W, b):
    """
        X: matriz de valores de las instancias (tamaño h x n) (instancias x features)
        W: matriz de pesos (tamaño n x k) (features x labels)
        b: vector de biases (tamaño K) (labels)
    """
    return np.dot(X,W) + b

# funcion para convertir la variable objetivo en un vector de 0s y 1s
def one_hot_encoding(label, classes):
    """
        la funcion genera un vector de 0s,
        busca la posicion del label en la lista de clases (dado por unique)
        y asigna 1 al vector en esa posicion
    """
    vector = np.zeros(len(classes))
    posicion = np.where(classes == label)[0][0]
    vector[posicion] = 1
    return vector

# funcion de costo usando los costos individuales
def loss_func(Y, Y_hat):
    """
        y:      valor real del label (0 or 1)
        y_hat:  valor predecido por el modelo (rango (0,1))
        h:      numero de instancias en el batch
        se calcula el promedio de error de todas las instancias
    """
    losses = 0
    for i in range(len(Y)):
        y = Y[i]
        y_hat = Y_hat[i]
        losses += cross_ent(y, y_hat)

    return (1/len(Y)) * losses

# representa el gradiente respecto a los logits (z antes de la funcion de activacion)
def E(Y, Y_hat):
    return Y_hat - Y

# funcion de gradiente para pesos y bias
def dW_db(X, dZ):
    dW = (1/len(X)) * np.dot(np.transpose(X), dZ)
    db = 0
    for i in range(len(X)):
        db += dZ[i]

    return  dW, (db/len(X))

def update(W, dW, b, db, alpha):
    W = W - alpha*dW
    b = b - alpha*db
    return W, b