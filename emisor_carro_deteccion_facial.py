import cv2
import tensorflow as tf
import numpy as np
import serial
import time

try:
    arduino = serial.Serial('COM15', 9600, timeout=1)
    time.sleep(2)
    print("Puerto abierto correctamente")
except Exception as e:
    print("Error:", e)
    arduino = None

# Cargar modelo CNN entrenado
cnn_model = tf.keras.models.load_model(r"C:\Escritorio_Resp_2\Proyectos_visual\Inteligencia_artificial\PIA\Entrenamiento\modelo_cnn.keras")

# Cargar clasificador de rostro de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

FRAMES_DORMIDO = 30
contador = 0
estado = "Despierto"
ultimo_estado_enviado = None  # Para controlar el envío al Arduino

cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo capturar frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            estado = "No se detecta rostro"
            contador = 0
        else:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (200, 200))
                face_img = face_img / 255.0
                face_img = np.expand_dims(face_img, axis=-1)
                face_img = np.expand_dims(face_img, axis=0)
                pred = cnn_model.predict(face_img)[0][0]
                estado_cnn = "Despierto" if pred < 0.5 else "Dormido"

                # Lógica temporal para alerta y comunicación serial
                if estado_cnn == "Dormido":
                    contador += 1
                    if contador >= FRAMES_DORMIDO:
                        estado = "DORMIDO! ALERTA!"
                        if arduino and ultimo_estado_enviado != "dormido":
                            arduino.write(b'1')
                            ultimo_estado_enviado = "dormido"
                else:
                    contador = 0
                    estado = "Despierto"
                    if arduino and ultimo_estado_enviado != "despierto":
                        arduino.write(b'0')
                        ultimo_estado_enviado = "despierto"
                cv2.putText(frame, f"{estado_cnn} ({pred:.2f})", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.putText(frame, f"Estado: {estado}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if estado == "Despierto" else (0, 0, 255), 2)
        cv2.putText(frame, f"Contador: {contador}/{FRAMES_DORMIDO}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if "ALERTA" in estado:
            cv2.putText(frame, "ALERTA: CONDUCTOR DORMIDO!", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow("Detección de Somnolencia CNN", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    print("Programa interrumpido por el usuario")

finally:
    cap.release()
    cv2.destroyAllWindows()
    if arduino:
        arduino.close()