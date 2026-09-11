import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, learning_curve
from xgboost import XGBClassifier


"""
    Funciones para el modelo a mano
"""
#funcion para separar el dataset en train y test
def split(X, Y, train_size=0.8):
    """
        X: matriz de features (instancias x features)
        Y: vector de labels (instancias)
        train_size: porcentaje de instancias que se usaran para entrenamiento
    """
    # numero de instancias
    m = len(X)
    # numero de instancias para entrenamiento
    train_size = int(m * train_size)

    # separar en train y test
    X_train = X[:train_size]
    Y_train = Y[:train_size]
    X_test = X[train_size:]
    Y_test = Y[train_size:]

    return X_train, Y_train, X_test, Y_test

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

# funcion para evaluar el desempeño del modelo
def performance(X_test, Y_test, W, b, classes):
    """
        model: funcion que recibe un vector de features y devuelve un vector de probabilidades
        se evalua el desempeño del modelo en el conjunto de test
    """
    Z_test = pred_func(X_test, W, b)
    Y_hat_test = g(Z_test)

    pred_indices = np.argmax(Y_hat_test, axis=1)
    pred_classes = classes[pred_indices]

    real_indices = np.argmax(Y_test, axis=1)
    real_classes = classes[real_indices]

    correct = 0
    for i in range(len(X_test)):
        if pred_classes[i] == real_classes[i]:
            correct += 1

    accuracy = correct / len(X_test)
    print(f"Accuracy: {accuracy*100:.2f}%")

"""
    Ahora para el modelo con framework
"""

# funcion para separar mis variables en categoricas y numericas
def split_features(data):
    cat = []
    num = []

    for column in data.columns:
        if data[column].dtype == "str":
            cat.append(column)
        else:
            num.append(column)

    return cat, num

# funcion para un gridsearch que encuentra hiperparametros optimos
def gs(num, cat):
    """
        num: features numericos
        cat: features categoricos
    """
    # para estandarizar features numericos y mantener features binarios
    preprocessor = ColumnTransformer(transformers=[("num", StandardScaler(), num), ("bin", "passthrough", cat)])

    # creo mi pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        # estare usando RF
        # ("model", RandomForestClassifier(random_state=67, class_weight="balanced", n_jobs=-1))
        # el "balanced" es muy importante porque el target esta desbalanceado
        ("model", XGBClassifier(random_state=67, eval_metric="mlogloss", n_jobs=-1))
    ])

    # para RF
    # param_grid = {
    #     "model__max_depth": [3, 5, 7, 10],
    #     "model__n_estimators": [100, 200],
    #     "model__max_features": ["sqrt", "log2"],
    #     "model__min_samples_leaf": [1, 2, 4]
    # }


    # para XGBoost
    param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.05, 0.1],
    "model__subsample": [0.8, 1.0]
}

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=5, n_jobs=-1,
        verbose = 1
    )

    return grid_search

def run_model(grid_search, X_train, y_train):
    model = grid_search.fit(X_train, y_train)
    print("Mejores parametros:")
    print(model.best_params_)

    print("Mejor f1_macro:")
    print(model.best_score_)

    best_model = model.best_estimator_

    return best_model

# funcion para guardar el modelo y no correrlo en cada compilacion
def save_model(model, encoder, path):
    joblib.dump({"model": model, "encoder": encoder}, path)

# funcion para cargar un modelo guardado
def load_model(path):
    data = joblib.load(path)
    return data["model"], data["encoder"]

def save_learning_curve(model, X_train, y_train, path):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=3, scoring="f1_macro",
        train_sizes=np.linspace(0.3, 1.0, 4), n_jobs=-1
    )

    data = {
        "train_sizes": train_sizes,
        "train_scores": train_scores,
        "val_scores": val_scores
    }

    joblib.dump(data, path)