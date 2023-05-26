import cv2
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

print("Librerias importadas")

cap = cv2.VideoCapture(1)  # Inicia la camara
previous_object_type = None  # variable para almacenar la figura detectada en la iteración anterior

# Inicializar Firebase
cred = credentials.Certificate('fir-ia-ca6df-firebase-adminsdk-kplvg-c00050cbaf.json')
firebase_admin.initialize_app(cred, {'databaseURL': 'https://fir-ia-ca6df-default-rtdb.firebaseio.com/'})
ref = db.reference('Robot')  # Actualiza 'nombre_del_nodo' con el nombre del nodo en tu base de datos

# Funcion para detectar esquinas
def getContours(img):
    global previous_object_type  # hacemos la variable global para poder modificarla desde dentro de la función
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # Encuentra los contornos
    for cnt in contours:
        area = cv2.contourArea(cnt)  # Encuentra el area de los contornos
        if area > 500:
            cv2.drawContours(frame, cnt, -1, (255, 0, 0), 3)  # Dibuja los contornos
            perimetro = cv2.arcLength(cnt, True)  # Encuentra el perimetro de los contornos
            aprrox = cv2.approxPolyDP(cnt, 0.02 * perimetro, True)  # Encuentra los vertices de los contornos
            objCorner = len(aprrox)  # Numero de vertices
            x, y, w, h = cv2.boundingRect(aprrox)  # Encuentra las coordenadas de los vertices

            if objCorner == 3:
                objectType = "Triangulo"
                current_object_type = 1
            elif objCorner == 4:
                aspRatio = w / float(h)
                if aspRatio > 0.95 and aspRatio < 1.05:
                    objectType = "Cuadrado"
                    current_object_type = 2
                else:
                    objectType = "Rectangulo"
                    current_object_type = 3
            elif objCorner > 4:
                objectType = "Circulo"
                current_object_type = 4
            else:
                objectType = "None"
                current_object_type = 5

            if current_object_type != previous_object_type:  # comparar con la figura detectada en la iteración anterior
                print(current_object_type)
                previous_object_type = current_object_type  # actualizar la figura anterior

                # Actualizar el valor en la base de datos
                ref.set(current_object_type)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, objectType, (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 0, 0), 2)

while True:
    ret, frame = cap.read()
    imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
    imgCanny = cv2.Canny(imgBlur, 50, 50)
    getContours(imgCanny)
    cv2.imshow("Figuras geometricas Stack", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
