import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np

# Rutas
train_dir = r"C:\Escritorio_Resp_2\Proyectos_visual\Inteligencia_artificial\PIA\Entrenamiento\Entrenamiento"
test_dir  = r"C:\Escritorio_Resp_2\Proyectos_visual\Inteligencia_artificial\PIA\Entrenamiento\Prueba"

# Preprocesamiento y generadores
img_size = (200, 200)
batch_size = 16

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',  # Cambia a 'categorical' si tienes más de 2 clases
    color_mode='grayscale',
    shuffle=True
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    color_mode='grayscale' ,
)

# Modelo CNN simple
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(200, 200, 1)),  # <-- Cambia a 1 canal
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenamiento
model.fit(
    train_gen,
    epochs=10,
    validation_data=test_gen
)

# Guardar modelo entrenado
model.save(r"C:\Escritorio_Resp_2\Proyectos_visual\Inteligencia_artificial\PIA\Entrenamiento\modelo_cnn.keras")# Puedes cambiar a .keras si prefieres

print("Modelo guardado como modelo_cnn.Keras")

# ----------- PREDICCIÓN DE UNA IMAGEN INDIVIDUAL -----------
# Puedes poner esto en otro archivo si lo prefieres

# Cargar modelo
model = tf.keras.models.load_model("modelo_cnn.keras")

# Cargar imagen de prueba
img_path = r"C:\Escritorio_Resp_2\Proyectos_visual\Inteligencia_artificial\PIA\Entrenamiento\Data\Models\dormido\Dormido_467.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (200, 200))
img = img / 255.0
img = np.expand_dims(img, axis=-1)
img = np.expand_dims(img, axis=0)

# Predecir
pred = model.predict(img)[0][0]
estado = "Despierto" if pred < 0.5 else "Dormido"
print(f"Predicción: {estado} (valor={pred:.2f})")