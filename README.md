# Deep-Learning-TC3009C.601

## Descripcion

Este proyecto consiste en la implementacion manual de un algoritmo de Machine Learnign, sin utilizar librerias o frameworks especializados en Machine Learning. Se permite el uso de herramientas como **NumPy, Pandas y Matplotlib** para operaciones matematicas, manipulacion de datos y visualizacion. El objetivo es comprender y desarrollar desde cero las diferentes etapas necesarias para entrenar y evaluar el modelo. Para este primer avance opte por tomar el aproach de **Machine Learning para clasificacion multiclase**.

## Dataset: Dry Bean

El dataset **Dry Bean** contiene informacion de **13,611 granos de frijol seco** pertenecientes a **7 variedades**:

* Seker
* Barbunya
* Bombay
* Cali
* Dermosan
* Horoz
* Sira

Cada muestra cuenta con **16 caracteristicas numericas** relacionadas con la geometria y forma del grano, obtenidas mediante procesamiento de imagenes.

| Caracteristica    | Descripcion                               |
| ----------------- | ----------------------------------------- |
| `Area`            | Area del grano en pixeles                 |
| `Perimeter`       | Perimetro del grano                       |
| `MajorAxisLength` | Longitud del eje principal                |
| `MinorAxisLength` | Longitud del eje menor                    |
| `AspectRatio`     | Relacion entre los ejes principal y menor |
| `Eccentricity`    | Excentricidad de la region                |
| `ConvexArea`      | Area convexa del grano                    |
| `EquivDiameter`   | Diametro equivalente                      |
| `Extent`          | Relacion entre el area y el bounding box  |
| `Solidity`        | Relacion entre el area y el area convexa  |
| `Roundness`       | Medida de redondez                        |
| `Compactness`     | Medida de compactacion                    |
| `ShapeFactor1`    | Factor de forma                           |
| `ShapeFactor2`    | Factor de forma                           |
| `ShapeFactor3`    | Factor de forma                           |
| `ShapeFactor4`    | Factor de forma                           |

La variable objetivo es `Class`, que corresponde a la variedad del frijol. El dataset **no contiene valores faltantes** ni **outliers**
