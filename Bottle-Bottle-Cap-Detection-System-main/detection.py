import cv2
from ultralytics import YOLO
import numpy as np

# Charger le modèle
model = YOLO('best.pt')

# Ouvrir la webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur: Impossible d'ouvrir la webcam")
    exit()

print("Appuyez sur 'q' pour quitter")

while True:
    # Lire une frame
    ret, frame = cap.read()
    
    if not ret:
        print("Erreur: Impossible de lire la frame")
        break
    
    # Transposer la matrice de la frame (échanger lignes et colonnes)
    frame_transposed = np.transpose(frame, (1, 0, 2))  # Transpose hauteur et largeur, garde les canaux couleur
    
    # Rendre la frame contiguë en mémoire pour le modèle
    frame_transposed = np.ascontiguousarray(frame_transposed)
    
    # Faire la détection sur la frame transposée
    results = model(frame_transposed, conf=0.5)
    
    # Dessiner les résultats sur la frame transposée
    annotated_frame = results[0].plot()
    
    # Afficher la frame
    cv2.imshow('Detection Bouteilles et Bouchons', annotated_frame)
    
    # Quitter si 'q' est pressé
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
