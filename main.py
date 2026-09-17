"""
    Juan Pablo Arteaga Patiño | A01665795
"""

# imports
import os

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder # para el xgboost

# aqui importo el resto de archivos
from cleaning import bean, tracks
from models import *
from graphics import *

MODEL_PATH = "modelo_xgboost.pkl"
LEARNING_CURVE_PATH = "learning_curve.pkl"

"""     MODELO 1 (Algoritmo manual)
"""

# separo features y target
X = bean.drop(columns=["Class"]).to_numpy(dtype=float)
Y = bean["Class"].to_numpy()

# normalizacion que implemente despues del primer testeo
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

classes = bean["Class"].unique()
Y_encoded = np.array([one_hot_encoding(label, classes) for label in Y])

# separo en train, test y validation con una particion de 0.7, 0.15, 0.15
X_train, Y_train, X_test, Y_test, X_val, Y_val = split(X, Y_encoded)

print("X shape:", X_train.shape)
print("Y shape:", Y_train.shape)
print("Classes:", classes)

# numero de features
n = X_train.shape[1]
# numero de clases
k = len(classes)

# pesos y bias iniciales
W = np.zeros((n, k))
b = np.zeros(k)

alpha = 0.01
epochs = 150
batch_size = 16

history_loss_train = []
history_loss_val = []

for epoch in range(epochs):

    # se recorre X en batches
    for i in range(0, len(X_train), batch_size):

        X_batch = X_train[i:i + batch_size]
        Y_batch = Y_train[i:i + batch_size]

        # prediccion
        Z = pred_func(X_batch, W, b)
        Y_hat = g(Z)

        # gradiente respecto a Z
        dZ = E(Y_batch, Y_hat)

        # gradientes de W y b
        dW, db = dW_db(X_batch, dZ)

        # actualizar parametros
        W, b = update(W, dW, b, db, alpha)

    # calculo loss para train
    Z_train = pred_func(X_train, W, b)
    Y_hat = g(Z_train)
    train_loss = loss_func(Y_train, Y_hat)
    # lo agrego al historico
    history_loss_train.append(train_loss)

    # calculo loss para val
    Z_val = pred_func(X_val, W, b)
    Y_hat_val = g(Z_val)
    val_loss = loss_func(Y_val, Y_hat_val)
    # lo agrego al historico
    history_loss_val.append(val_loss)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: train_loss = {train_loss}, val_loss = {val_loss}")

# prediccion
Z_test = pred_func(X_test, W, b)
Y_hat_test = g(Z_test)

# se toma la clase con mayor probabilidad
pred_indices = np.argmax(Y_hat_test, axis=1)
pred_classes = classes[pred_indices]

real_indices = np.argmax(Y_test, axis=1)
real_classes = classes[real_indices]

print("\nPredicciones:")
for i in range(15):
    print(
        f"Prediccion: {pred_classes[i]} | "
        f"Real: {real_classes[i]} | "
        # f"Probabilidades: {np.round(Y_hat_test[i], 2)}"
    )

cm1 = compute_confusion_matrix(real_indices, pred_indices, k)

# evaluar el modelo y plot
performance(Y_hat_test, pred_indices, real_indices, real_classes)
plot_loss(history_loss_train, history_loss_val)
conf_mat(cm1, classes)

"""     MODELO 2 (Algoritmo con Frameworks)
"""

X = tracks.drop(columns=["track_genre_top"])
Y = tracks["track_genre_top"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=67)

cat_cols, num_cols = split_features(X_train)

if os.path.exists(MODEL_PATH):
    print("Cargando modelo.............")

    model, encoder = load_model(MODEL_PATH)
    y_test_encoded = encoder.transform(y_test)


else:
    print("Entrenando modelo..........")

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    estimator, depth = 500, 5
    pipeline = build_pipeline(num_cols, cat_cols, estimator, depth)
    model = run_model(pipeline, X_train, y_train_encoded, estimator, depth)

    save_model(model, encoder, MODEL_PATH)
    print("Modelo guardado.")

y_train_encoded = encoder.transform(y_train)
y_pred = model.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
# f1_macro = f1_score(y_test, y_pred, average="macro")
# f1_weighted = f1_score(y_test, y_pred, average="weighted")
# cm2 = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test_encoded, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test_encoded, y_pred)
f1_macro = f1_score(y_test_encoded, y_pred, average="macro")
f1_weighted = f1_score(y_test_encoded, y_pred, average="weighted")

cm2 = confusion_matrix(y_test_encoded, y_pred)

print(f"Accuracy:          {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"F1 Macro:          {f1_macro:.4f}")
print(f"F1 Weighted:       {f1_weighted:.4f}")

classes = encoder.classes_

print(classification_report(y_test_encoded, y_pred, target_names=classes))
# print(classification_report(y_test, y_pred))

# plots 
target_histogram(Y,classes)
conf_mat(cm2, classes)
conf_mat_norm(cm2, classes)
roc_curve_plot(model, X_test, y_test, classes)

if os.path.exists(LEARNING_CURVE_PATH):
    load_learning_curve(LEARNING_CURVE_PATH)
else:
    save_learning_curve(model, X_train, y_train_encoded, LEARNING_CURVE_PATH)
    load_learning_curve(LEARNING_CURVE_PATH)