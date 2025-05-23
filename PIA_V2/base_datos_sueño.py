import cv2
import os
import imutils

etiquetname = "despierto"
datapath = "C:/Escritorio_Resp_2/Proyectos_visual/Inteligencia_artificial/PIA/Entrenamiento/Despiertos"
etiquetpath = datapath + "/" + etiquetname

if not os.path.exists(etiquetpath):
    print("carpeta creada:", etiquetpath)
    os.makedirs(etiquetpath)
#"videos_dormido/itzel.mp4"
cap = cv2.VideoCapture("C:/Escritorio_Resp_2/Proyectos_visual/Inteligencia_artificial/PIA/Entrenamiento/videos_despierto/belen.mp4")
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
count = 786

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    if not cap.isOpened():
        print("No se pudo abrir el video.")
        exit()

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        despierto = auxFrame[y:y + h, x:x + w]
        despierto = cv2.resize(despierto, (200, 200), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(f"{etiquetpath}/Despierto_{count}.jpg", despierto)
        count += 1

    cv2.imshow("frame", frame)

    k = cv2.waitKey(1)
    if k == 27 or count >= 1058:
        break

cap.release()
cv2.destroyAllWindows()