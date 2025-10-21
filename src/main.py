import cv2
import mediapipe as mp
from utils import draw_face_mesh  # Asegúrate que esta función exista en utils.py

# Inicializar los módulos
mp_face_mesh = mp.solutions.face_mesh

# Configurar la cámara
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo acceder a la cámara")
            break

        # Convertir a RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar los rostros
        results = face_mesh.process(rgb_frame)

        # Dibujar el malla facial
        if results.multi_face_landmarks:
            frame = draw_face_mesh(frame, results)

        # Mostrar en pantalla
        cv2.imshow("FaceMesh", frame)

        # Salir con tecla ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
