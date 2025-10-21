import cv2
import mediapipe as mp

# Inicializamos los módulos de dibujo y la malla facial
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

def init_camera(index=0, width=1280, height=720):
    """
    Inicializa la cámara con el índice especificado.
    :param index: Índice de la cámara (por defecto 0)
    :param width: Ancho del frame
    :param height: Alto del frame
    :return: Objeto VideoCapture de OpenCV
    """
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    if not cap.isOpened():
        print("⚠️ No se pudo abrir la cámara.")
    else:
        print("✅ Cámara inicializada correctamente.")
    
    return cap


def draw_face_mesh(frame, results):
    """
    Dibuja la malla facial sobre el frame original.
    :param frame: Imagen original (BGR)
    :param results: Resultados del procesamiento de MediaPipe
    :return: Frame con la malla dibujada
    """
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Dibuja la malla facial con estilo personalizado
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0),  # Verde
                    thickness=1,
                    circle_radius=1
                )
            )
            # Opcional: dibujar contornos de ojos y labios
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(255, 0, 0),  # Azul
                    thickness=1,
                    circle_radius=1
                )
            )
    return frame
