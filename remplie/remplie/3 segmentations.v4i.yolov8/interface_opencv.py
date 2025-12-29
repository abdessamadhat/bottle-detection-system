#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface OpenCV Simple pour le Modèle YOLOv8 Segmentation
Détection et segmentation de bouteilles en temps réel
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
from pathlib import Path
import time

# Assurer que le script s'exécute dans le bon répertoire
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"🔄 Répertoire de travail: {os.getcwd()}")

class BottleSegmentationInterface:
    def __init__(self, model_path=None):
        """
        Initialise l'interface de segmentation
        
        Args:
            model_path: Chemin vers le modèle .pt (None = utilise le meilleur disponible)
        """
        self.model_path = model_path
        self.model = None
        self.class_names = {0: 'bottle', 1: 'water'}
        self.colors = {
            0: (0, 255, 0),    # Vert pour bouteille
            1: (255, 0, 0)     # Bleu pour eau
        }
        
        # Statistiques
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        self.load_model()
    
    def find_best_model(self):
        """Trouve le meilleur modèle disponible"""
        possible_paths = [
            "runs/segment/bottle_final_quality/weights/best.pt",
            "runs/segment/bottle_final_quality/weights/last.pt",
            "runs/segment/bottle_quality_offline/weights/best.pt", 
            "runs/segment/bottle_quality_offline/weights/last.pt",
            "runs/segment/bottle_water_v2/weights/best.pt",
            "yolov8m-seg.pt"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Modèle trouvé: {path}")
                return path
        
        print("❌ Aucun modèle trouvé!")
        return None
    
    def load_model(self):
        """Charge le modèle YOLOv8"""
        if self.model_path is None:
            self.model_path = self.find_best_model()
        
        if self.model_path and os.path.exists(self.model_path):
            try:
                self.model = YOLO(self.model_path)
                print(f"🚀 Modèle chargé: {self.model_path}")
                
                # Test GPU
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                print(f"🔧 Device: {device}")
                
                return True
            except Exception as e:
                print(f"❌ Erreur chargement modèle: {e}")
                return False
        else:
            print(f"❌ Modèle introuvable: {self.model_path}")
            return False
    
    def draw_predictions(self, frame, results):
        """
        Dessine les prédictions sur l'image
        
        Args:
            frame: Image OpenCV
            results: Résultats YOLO
            
        Returns:
            Image annotée
        """
        annotated_frame = frame.copy()
        
        if results and len(results) > 0:
            result = results[0]
            
            # Dessiner les masques de segmentation
            if hasattr(result, 'masks') and result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                
                for i, mask in enumerate(masks):
                    # Redimensionner le masque à la taille de l'image
                    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)
                    
                    # Obtenir la classe et la couleur
                    class_id = int(result.boxes.cls[i].cpu().numpy())
                    confidence = float(result.boxes.conf[i].cpu().numpy())
                    color = self.colors.get(class_id, (255, 255, 255))
                    
                    # Créer une overlay colorée pour le masque
                    overlay = annotated_frame.copy()
                    overlay[mask_binary == 1] = color
                    
                    # Mélanger avec l'image originale (transparence)
                    cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)
                    
                    # Dessiner le contour du masque
                    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(annotated_frame, contours, -1, color, 2)
            
            # Dessiner les bounding boxes
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                for box, conf, cls in zip(boxes, confidences, classes):
                    x1, y1, x2, y2 = map(int, box)
                    class_id = int(cls)
                    class_name = self.class_names.get(class_id, f"Class {class_id}")
                    color = self.colors.get(class_id, (255, 255, 255))
                    
                    # Rectangle
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Label avec confiance
                    label = f"{class_name}: {conf:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame
    
    def draw_interface_info(self, frame):
        """Dessine les informations de l'interface"""
        h, w = frame.shape[:2]
        
        # Calculer FPS
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.fps = self.frame_count / elapsed
        
        # Fond semi-transparent pour les infos
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Texte d'informations
        infos = [
            f"FPS: {self.fps:.1f}",
            f"Modele: {os.path.basename(self.model_path) if self.model_path else 'N/A'}",
            f"Resolution: {w}x{h}",
            "Controles:"
        ]
        
        for i, info in enumerate(infos):
            cv2.putText(frame, info, (20, 35 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Contrôles
        controls = [
            "ESC/Q: Quitter",
            "SPACE: Pause",
            "S: Screenshot"
        ]
        
        for i, control in enumerate(controls):
            cv2.putText(frame, control, (20, 95 + i * 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    def webcam_inference(self):
        """Lance l'inférence en temps réel avec webcam"""
        if not self.model:
            print("❌ Modèle non chargé!")
            return
        
        print("🔄 Initialisation de la webcam...")
        
        # Essayer plusieurs indices de caméra
        cap = None
        for camera_index in [0, 1, 2]:
            print(f"   Essai caméra index {camera_index}...")
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                print(f"✅ Webcam trouvée sur l'index {camera_index}")
                break
            cap.release()
        
        if cap is None or not cap.isOpened():
            print("❌ Impossible d'ouvrir la webcam!")
            print("💡 Vérifiez que votre caméra n'est pas utilisée par une autre application")
            return
        
        # Configuration webcam
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("🎥 Webcam démarrée - Appuyez sur ESC ou Q pour quitter")
        print("🖼️ Ouverture de la fenêtre d'interface...")
        
        paused = False
        window_name = 'YOLOv8 Bottle Segmentation - Webcam'
        
        # Créer la fenêtre et la positionner
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO)
        cv2.moveWindow(window_name, 100, 100)
        
        # Forcer l'affichage de la fenêtre
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Erreur lecture webcam")
                    break
                
                # Prédiction
                results = self.model(frame, conf=0.3, iou=0.5)
                
                # Annotation
                annotated_frame = self.draw_predictions(frame, results)
            
            # Interface
            self.draw_interface_info(annotated_frame if not paused else frame)
            
            # Affichage
            cv2.imshow('YOLOv8 Bottle Segmentation - Webcam', annotated_frame if not paused else frame)
            
            # Contrôles clavier
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC ou Q
                break
            elif key == ord(' '):  # SPACE
                paused = not paused
                print(f"{'⏸️ Pause' if paused else '▶️ Reprise'}")
            elif key == ord('s'):  # S pour screenshot
                filename = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, annotated_frame if not paused else frame)
                print(f"📸 Screenshot sauvegardé: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Interface fermée")
    
    def image_inference(self, image_path):
        """
        Lance l'inférence sur une image
        
        Args:
            image_path: Chemin vers l'image
        """
        if not self.model:
            print("❌ Modèle non chargé!")
            return
        
        if not os.path.exists(image_path):
            print(f"❌ Image introuvable: {image_path}")
            return
        
        # Charger image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ Impossible de charger: {image_path}")
            return
        
        print(f"🖼️ Analyse de: {image_path}")
        
        # Prédiction
        results = self.model(frame, conf=0.3, iou=0.5)
        
        # Annotation
        annotated_frame = self.draw_predictions(frame, results)
        
        # Interface info
        self.draw_interface_info(annotated_frame)
        
        # Affichage
        window_name = f'YOLOv8 Bottle Segmentation - {os.path.basename(image_path)}'
        cv2.imshow(window_name, annotated_frame)
        
        print("🎯 Appuyez sur une touche pour fermer ou S pour sauvegarder")
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('s'):
            output_path = f"result_{os.path.basename(image_path)}"
            cv2.imwrite(output_path, annotated_frame)
            print(f"💾 Résultat sauvegardé: {output_path}")
        
        cv2.destroyAllWindows()


def main():
    """Lancement automatique de l'interface webcam"""
    print("=" * 60)
    print("🍶 INTERFACE YOLOV8 BOTTLE SEGMENTATION")
    print("=" * 60)
    print("🚀 Démarrage automatique en mode webcam...")
    
    try:
        # Test OpenCV
        print("🔍 Vérification d'OpenCV...")
        print(f"OpenCV version: {cv2.__version__}")
        
        # Initialiser l'interface
        print("🔄 Initialisation de l'interface...")
        interface = BottleSegmentationInterface()
        
        if not interface.model:
            print("❌ Impossible de charger un modèle!")
            print("💡 Assurez-vous qu'un modèle .pt existe dans runs/segment/")
            input("Appuyez sur Entrée pour quitter...")
            return
        
        # Test de la webcam
        print("📹 Test d'accès à la webcam...")
        test_cap = cv2.VideoCapture(0)
        if not test_cap.isOpened():
            print("❌ Webcam non accessible!")
            print("💡 Vérifiez que votre caméra est connectée et disponible")
            input("Appuyez sur Entrée pour quitter...")
            return
        test_cap.release()
        print("✅ Webcam accessible")
        
        # Lancer directement la webcam
        print("🎥 Lancement de la webcam en temps réel...")
        print("🔥 Appuyez sur 'q' ou 'ESC' pour quitter l'application")
        print("🖼️ La fenêtre d'interface va s'ouvrir...")
        
        interface.webcam_inference()
        
        print("👋 Interface fermée!")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        input("Appuyez sur Entrée pour quitter...")


if __name__ == "__main__":
    main()