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
    """
        y es un vector de 0s y 1s
        y_hat es un vector en el rango (0,1)
    """
    # me encontre con warnings de precision numerica de float asi que uso .clip
    y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15) # se asegura que ningun valor sea menor o mayor a 1e-15 
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

"""
    ENTREAMIENTO 1
    Me enfrenté con problemas de punto flotante y poca tasa de aprendizaje.
    Para solucionarlo, modifiqué el código de cross entropy, añadí el .clip
    también agregué una estandarización de datos para que todos los features
    aprendieran al mismo ritmo, pues su varianza es muy distinta.
"""

"""
    ENTRENAMIENTO 2:
"""

# como el dataset originalmente está ordenado por clases, lo randomizo
bean = bean.sample(frac=1, random_state=67).reset_index(drop=True)

# separo features y target
X = bean.drop(columns=["Class"]).to_numpy(dtype=float)
Y = bean["Class"].to_numpy()

# normalizacion que implemente despues del primer testeo
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

classes = bean["Class"].unique()
Y_encoded = np.array([
    one_hot_encoding(label, classes)
    for label in Y
])

print("X shape:", X.shape)
print("Y shape:", Y_encoded.shape)
print("Classes:", classes)

# numero de features
n = X.shape[1]
# numero de clases
k = len(classes)

# pesos y bias iniciales
W = np.zeros((n, k))
b = np.zeros(k)

alpha = 0.01
epochs = 100
batch_size = 16

for epoch in range(epochs):

    # se recorre X en batches
    for i in range(0, len(X), batch_size):

        X_batch = X[i:i + batch_size]
        Y_batch = Y_encoded[i:i + batch_size]

        # prediccion
        Z = pred_func(X_batch, W, b)
        Y_hat = g(Z)

        # gradiente respecto a Z
        dZ = E(Y_batch, Y_hat)

        # gradientes de W y b
        dW, db = dW_db(X_batch, dZ)

        # actualizar parametros
        W, b = update(W, dW, b, db, alpha)

    # calculo loss
    Z = pred_func(X, W, b)
    Y_hat = g(Z)

    loss = loss_func(Y_encoded, Y_hat)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss = {loss}")

# tomamos algunas instancias para evaluar nuestro modelo
X_test = X[:5]
# prediccion
Z_test = pred_func(X_test, W, b)
Y_hat_test = g(Z_test)

# se toma la clase con mayor probabilidad
pred_indices = np.argmax(Y_hat_test, axis=1)
pred_classes = classes[pred_indices]

print("\nPredicciones:")
for i in range(len(X_test)):
    print(
        f"Prediccion: {pred_classes[i]} | "
        f"Real: {Y[i]} | "
        f"Probabilidades: {np.round(Y_hat_test[i], 2)}"
    )
