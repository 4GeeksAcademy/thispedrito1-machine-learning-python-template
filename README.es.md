<!-- hide -->

# RNA para clasificación de imágenes - Guía paso a paso

<!-- endhide -->

- Comprender un dataset nuevo.
- Modelar los datos utilizando una RNA.
- Analizar los resultados y optimizar el modelo.

## 🌱 Cómo iniciar este proyecto

Sigue las siguientes instrucciones:

1. Crea un nuevo repositorio basado en el [proyecto de Machine Learning](https://github.com/4GeeksAcademy/machine-learning-python-template) o [haciendo clic aquí](https://github.com/4GeeksAcademy/machine-learning-python-template/generate).
2. Abre el repositorio creado recientemente en Codespace usando la [extensión del botón de Codespace](https://docs.github.com/en/codespaces/developing-in-codespaces/creating-a-codespace-for-a-repository#creating-a-codespace-for-a-repository).
3. Una vez que el VSCode del Codespace haya terminado de abrirse, comienza tu proyecto siguiendo las instrucciones a continuación.

## 🚛 Cómo entregar este proyecto

Una vez que hayas terminado de resolver el caso práctico, asegúrate de confirmar tus cambios, haz push a tu repositorio y ve a 4Geeks.com para subir el enlace del repositorio.

## 📝 Instrucciones

### Sistema de clasificación de imágenes

El conjunto de datos se compone de fotos de perros y gatos proporcionadas como un subconjunto de fotos de uno mucho más grande de 3 millones de fotos anotadas manualmente. Estos datos se obtuvieron a través de una colaboración entre Petfinder.com y Microsoft.

El conjunto de datos se usó originalmente como un CAPTCHA, es decir, una tarea que se cree que un humano encuentra trivial, pero que una máquina no puede resolver, que se usa en sitios web para distinguir entre usuarios humanos y bots. La tarea se denominó "Asirra". Cuando se presentó "Asirra", se mencionó "que los estudios de usuarios indican que los humanos pueden resolverlo el 99,6% de las veces en menos de 30 segundos". A menos que se produzca un gran avance en la visión artificial, esperamos que los ordenadores no tengan más de 1/54.000 posibilidades de resolverlo.

En el momento en que se publicó la competencia, el resultado de última generación se logró con un SVM y se describió en un artículo de 2007 con el título "Ataques de Machine Learning contra el CAPTCHA de Asirra" (PDF) que logró una precisión de clasificación del 80%. Fue este documento el que demostró que la tarea ya no era una tarea adecuada para un CAPTCHA poco después de que se propusiera la tarea.

#### Paso 1: Carga del conjunto de datos

El conjunto de datos se encuentra en este [link](https://storage.googleapis.com/datascience-materials/dogs-vs-cats.zip). Descarga la carpeta y descomprime los archivos. Ahora tendrás una carpeta con el dataset y una carpeta llamada `train` que contiene más de 25.000 archivos de imagen (formato .jpg) de perros y gatos. Las fotos están etiquetadas por su nombre de archivo, con la palabra `dog` o `cat`.

#### Paso 2: Visualiza la información de entrada

El primer paso cuando nos enfrentamos a un problema de clasificación de imágenes es obtener toda la información posible a través de ellas. Por lo tanto, carga e imprime las primeras nueve fotos de perros en una sola figura. Repite lo mismo para los gatos. Puedes ver que las fotos son a color y tienen diferentes formas y tamaños.

Esta variedad de tamaños y formatos debe solucionarse antes de entrenar el modelo. Asegúrate de que todas tengan un tamaño fijo de 200x200 píxeles.

Como podrás ver, son una gran cantidad de imágenes, asegúrate de seguir las siguientes normas:

1. **Si tienes más de 12 gigabytes de RAM**, usa la API de procesamiento de imágenes de Keras para cargar las 25.000 fotos en el conjunto de datos de entrenamiento y remodelarlas a fotos cuadradas de 200×200 píxeles. La etiqueta también debe determinarse para cada foto en función de los nombres de archivo. Se debe guardar una tupla de fotos y etiquetas.
2. **Si no tienes más de 12 gigabytes de RAM**, carga las imágenes progresivamente usando la clase Keras `ImageDataGenerator` y la función `flow_from_directory()`. Esto será más lento de ejecutar, pero se ejecutará en hardware de menor capacidad. Esta función prefiere que los datos se dividan en directorios _train_ y _test_ separados, y debajo de cada directorio para tener un subdirectorio para cada clase.

Una vez tengas todas las imágenes procesadas, crea un objeto `ImageDataGenerator` para datos de entrenamiento y prueba. Luego pasa la carpeta que tiene datos de entrenamiento al objeto `trdata` y, de manera similar, pasa la carpeta que tiene datos de prueba al objeto `tsdata`. De esta forma, se etiquetarán las imágenes automáticamente y estará todo listo para entrar a la red.

#### Paso 3: Construye una RNA

Cualquier clasificador que se ajuste a este problema tendrá que ser robusto porque algunas imágenes muestran al gato o al perro en una esquina o tal vez a 2 gatos o perros en la misma foto. Si has podido investigar algunas de las implementaciones de los ganadores de otras competiciones también relacionadas con imágenes, verás que `EfficientNet-B0` es una arquitectura de CNN presentada por Google en 2019. Usa un escalado compuesto de profundidad, anchura y resolución para alcanzar una gran precisión en ImageNet con muchos menos parámetros que modelos anteriores, y sigue siendo una base sólida y eficiente para tareas de visión.

Utiliza la siguiente arquitectura de prueba:

```py
from keras.applications import EfficientNetB0
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D

model = Sequential()
model.add(EfficientNetB0(include_top = False, weights = None, input_shape = (224, 224, 3)))
model.add(GlobalAveragePooling2D())
model.add(Dense(units = 128, activation = "relu"))
model.add(Dense(units = 2, activation = "softmax"))
```

El código anterior carga la backbone convolucional de `EfficientNet-B0`, reduce sus mapas de características con `GlobalAveragePooling2D` y después aplica capas densas (capas `Dense`) para la clasificación final de perro/gato.

A continuación añade los elementos restantes para conformar el modelo, entrénalo y mide su rendimiento.

#### Paso 4: Optimiza el modelo anterior

Importa el método `ModelCheckpoint` y `EarlyStopping` de Keras. Crea un objeto de ambos y pásalo como funciones callback a `fit_generator`.

Carga el mejor modelo de los anteriores y utiliza el conjunto de test para hacer predicciones.

#### Paso 5: Guarda el modelo

Almacena el modelo en la carpeta correspondiente.

> Nota: También incorporamos muestras de solución en [este link](https://github.com/4GeeksAcademy/image-classifier-project-tutorial/blob/main/solution.es.ipynb) que te sugerimos honestamente que solo uses si estás atascado por más de 30 minutos o si ya has terminado y quieres compararlo con tu enfoque.
