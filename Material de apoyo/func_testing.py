from main import *

# test de g
test_z = np.array([[3, 3, 3]])
print(g(test_z))
# deberian ser tres valores iguales y sumar 1


# test de la funcion de cross-entropy
test_y = np.array([0, 1, 0, 0, 0, 0, 0])
test_y_hat = np.array([0.1, 0.7, 0.1, 0.05, 0.02, 0.02, 0.01])
print(cross_ent(test_y, test_y_hat), -np.log(0.7))
# deberian ser iguales ambos valores


# test de la funcion de prediccion
test_X = np.array([
    [0.6, 0.1, 0.05],
    [0.4, 0.2, 0.9],
    [0.08, 0.8, 0.9],
    [0.2, 0.4, 0.6]
])

test_W = np.array([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6]
])

test_b = np.array([0.1, 0.2])

print(pred_func(test_X, test_W, test_b))
# deberia imprimir una matriz de tamaño (4, 2)


# test de one-hot encoding
test_classes = np.array([0, 1, 2, 3, 4, 5, 6])
test_label = 3

print(one_hot_encoding(test_label, test_classes))
# deberia imprimir:
# [0 0 0 1 0 0 0]


# test de la funcion de costo para un batch
test_Y = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

test_Y_hat = np.array([
    [0.8, 0.1, 0.1],
    [0.2, 0.7, 0.1],
    [0.1, 0.2, 0.7]
])

print(loss_func(test_Y, test_Y_hat))
# deberia ser igual a:
print((-np.log(0.8) - np.log(0.7) - np.log(0.7)) / 3)


# test de E (gradiente respecto a logits)
test_Y = np.array([
    [1, 0, 0],
    [0, 1, 0]
])

test_Y_hat = np.array([
    [0.8, 0.1, 0.1],
    [0.2, 0.7, 0.1]
])

print(E(test_Y, test_Y_hat))


# test de dW_db
test_X = np.array([
    [1, 2],
    [3, 4]
])

test_dZ = np.array([
    [ 0.1, -0.1],
    [-0.2,  0.2]
])

test_dW, test_db = dW_db(test_X, test_dZ)

print("dW:")
print(test_dW)

print("db:")
print(test_db)


# test de update
test_W = np.array([
    [1.0, 2.0],
    [3.0, 4.0]
])

test_b = np.array([1.0, 2.0])

test_dW = np.array([
    [0.1, 0.2],
    [0.3, 0.4]
])

test_db = np.array([0.1, 0.2])

test_alpha = 0.5

new_W, new_b = update(
    test_W,
    test_dW,
    test_b,
    test_db,
    test_alpha
)

print("W actualizado:")
print(new_W)

print("b actualizado:")
print(new_b)