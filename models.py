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
def split(X, Y, train_size=0.7, val_size=0.15):
    """
    X: matriz de features
    Y: matriz de labels one-hot
    train_size: porcentaje para entrenamiento
    val_size: porcentaje para validación
    """

    m = len(X)

    train_end = int(m * train_size)
    val_end = int(m * (train_size + val_size))

    X_train = X[:train_end]
    Y_train = Y[:train_end]

    X_val = X[train_end:val_end]
    Y_val = Y[train_end:val_end]

    X_test = X[val_end:]
    Y_test = Y[val_end:]

    return X_train, Y_train, X_test, Y_test, X_val, Y_val

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
def performance(yhat, pred_ind, real_ind, real_class):
    """
        Evalúa el desempeño del modelo usando
        Cross Entropy Loss, F1 Macro, F1 por cada clase
        yhat       : probabilidades predichas
        pred_ind   : índices de las clases predichas
        real_ind   : índices de las clases reales
        real_class : clases reales
    """

    # número de clases
    n_classes = yhat.shape[1]

    # reconstruir Y en one-hot
    y = np.zeros_like(yhat)
    y[np.arange(len(real_ind)), real_ind] = 1

    # Cross-Entropy Loss
    loss = loss_func(y, yhat)

    # matriz de confusión
    confusion = np.zeros((n_classes, n_classes), dtype=int)

    for real, pred in zip(real_ind, pred_ind):
        confusion[real, pred] += 1

    # F1 por clase
    f1_per_class = np.zeros(n_classes)

    for i in range(n_classes):

        TP = confusion[i, i]
        FP = np.sum(confusion[:, i]) - TP
        FN = np.sum(confusion[i, :]) - TP

        if TP + FP > 0:
            precision = TP / (TP + FP)
        else:
            precision = 0

        if TP + FN > 0:
            recall = TP / (TP + FN)
        else:
            recall = 0

        if precision + recall > 0:
            f1_per_class[i] = (
                2 * precision * recall
                / (precision + recall)
            )

    # Macro F1
    macro_f1 = np.mean(f1_per_class)

    # resultados
    print(f"Cross-Entropy Loss: {loss:.4f}")
    print(f"Macro F1:           {macro_f1:.4f}")

    print("\nF1 por clase:")
    for clase, f1 in zip(np.unique(real_class), f1_per_class):
        print(f"{clase}: {f1:.4f}")

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

# funcion para crear el pipeline con hiperparametros fijados
def build_pipeline(num, cat, estimator, depth):
    """
        num: features numericos
        cat: features categoricos
        
        Construye un pipeline con XGBoost usando hiperparametros fijados manualmente.
        Los hiperparametros fueron seleccionados basandose en validacion cruzada anterior.
    """
    # para estandarizar features numericos y mantener features binarios
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num), 
            ("bin", "passthrough", cat)
        ]
    )

    # creo mi pipeline con hiperparametros fijados
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(
            # los voy modificando manualmente
            n_estimators=estimator,
            max_depth=depth,
            learning_rate=0.1,
            subsample=0.8,
            random_state=67,
            eval_metric="mlogloss",
            n_jobs=-1
        ))
    ])

    return pipeline

def run_model(pipeline, X_train, y_train, estimator, depth):
    """
        Entrena el modelo usando el pipeline preconfigurado.
    """
    model = pipeline.fit(X_train, y_train)
    print("Hiperparametros del modelo:")
    print(f"  n_estimators: {estimator}")
    print(f"  max_depth: {depth}")
    print(f"  learning_rate: 0.1")
    print(f"  subsample: 0.8")
    print("Modelo entrenado exitosamente.")

    return model

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