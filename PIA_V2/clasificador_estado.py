import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

datapath = r"C:\Escritorio_Resp_2\Proyectos_visual\Inteligencia_artificial\PIA\Entrenamiento\Data\Models"
etiquetname  = os.listdir(datapath)
print("Lista de Estados:", etiquetname)

labels = []
EstadoData = []
file_names = []
label = 0

for nameDir in etiquetname:
    etiquetpath = datapath + "/" + nameDir
    print("leyendo las imagenes")

    for fileName in os.listdir(etiquetpath):
        if fileName.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image = cv2.imread(etiquetpath + "/" + fileName, 0)
            if image is not None:
                labels.append(label)
                EstadoData.append(image)
                file_names.append(fileName)  # Guarda el nombre original
    label += 1

EstadoData = np.array(EstadoData)
labels = np.array(labels)
file_names = np.array(file_names)

# Divide en entrenamiento y prueba, incluyendo los nombres originales
X_train, X_test, y_train, y_test, fn_train, fn_test = train_test_split(
    EstadoData, labels, file_names, test_size=0.2, random_state=42, stratify=labels
)

output_train_path = "C:/Escritorio_Resp_2/Proyectos_visual/Inteligencia_artificial/PIA/Entrenamiento/Entrenamiento"
output_test_path  = "C:/Escritorio_Resp_2/Proyectos_visual/Inteligencia_artificial/PIA/Entrenamiento/Prueba"

os.makedirs(os.path.join(output_train_path, "Despierto"), exist_ok=True)
os.makedirs(os.path.join(output_train_path, "Dormido"), exist_ok=True)
os.makedirs(os.path.join(output_test_path, "Despierto"), exist_ok=True)
os.makedirs(os.path.join(output_test_path, "Dormido"), exist_ok=True)

# Guardar imágenes de entrenamiento con nombre original
for img, label, fname in zip(X_train, y_train, fn_train):
    class_name = "Despierto" if label == 0 else "Dormido"
    cv2.imwrite(f"{output_train_path}/{class_name}/{fname}", img)

# Guardar imágenes de prueba con nombre original
for img, label, fname in zip(X_test, y_test, fn_test):
    class_name = "Despierto" if label == 0 else "Dormido"
    cv2.imwrite(f"{output_test_path}/{class_name}/{fname}", img)

# Divide en entrenamiento y prueba, incluyendo los nombres originales
X_train, X_test, y_train, y_test, fn_train, fn_test = train_test_split(
    EstadoData, labels, file_names, test_size=0.2, random_state=42, stratify=labels
)

output_train_path = "C:/Escritorio_Resp_2/Proyectos_visual/Inteligencia_artificial/PIA/Entrenamiento/Entrenamiento"
output_test_path  = "C:/Escritorio_Resp_2/Proyectos_visual/Inteligencia_artificial/PIA/Entrenamiento/Prueba"

os.makedirs(output_train_path, exist_ok=True)
os.makedirs(output_test_path, exist_ok=True)

# Guardar imágenes de entrenamiento mezcladas
for img, label, fname in zip(X_train, y_train, fn_train):
    class_name = "Despierto" if label == 0 else "Dormido"
    cv2.imwrite(f"{output_train_path}/{class_name}_{fname}", img)

# Guardar imágenes de prueba mezcladas
for img, label, fname in zip(X_test, y_test, fn_test):
    class_name = "Despierto" if label == 0 else "Dormido"
    cv2.imwrite(f"{output_test_path}/{class_name}_{fname}", img)