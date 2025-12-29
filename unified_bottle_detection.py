#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface Unifiée - Détection et Segmentation de Bouteilles
Affichage simultané des 2 modèles dans une seule fenêtre OpenCV
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
from pathlib import Path
import time
from datetime import datetime

class UnifiedBottleDetection:
    def __init__(self):
        """Initialise l'interface unifiée avec les 2 modèles"""
        print("=" * 70)
        print("🍶 UNIFIED BOTTLE DETECTION - DOUBLE MODEL DISPLAY")
        print("=" * 70)
        
        # === MODÈLE 1: DÉTECTION (Bouteille + Bouchon) ===
        self.model_detection_path = 'Bottle-Bottle-Cap-Detection-System-main/best.pt'
        self.model_detection = None
        self.detection_classes = {0: 'Bottle', 1: 'Avec Bouchon', 2: 'Sans Bouchon'}
        self.detection_colors = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 165, 0)}  # Vert, Rouge, Orange
        
        # === MODÈLE 2: SEGMENTATION (Bouteille + Eau) ===
        self.model_segmentation_path = 'remplie/remplie/3 segmentations.v4i.yolov8/runs/segment/bottle_final_quality/weights/best.pt'
        self.model_segmentation = None
        self.segmentation_classes = {0: 'Bottle', 1: 'Water'}
        self.segmentation_colors = {0: (0, 255, 0), 1: (255, 0, 0)}  # Vert, Bleu
        
        # === PARAMÈTRES ===
        self.confidence_detection = 0.5
        self.confidence_segmentation = 0.3
        self.iou = 0.5
        self.transpose_image = True  # Transpose activé par défaut
        self.img_size = 320  # Taille très réduite pour inférence plus rapide
        self.process_every_n_frames = 2  # Traiter 1 frame sur 2 pour chaque modèle
        self.frame_skip_counter = 0
        
        # === STATISTIQUES ===
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # === DÉTECTION DE REMPLISSAGE ===
        self.water_percentage_history = []  # Historique des pourcentages
        self.history_size = 10  # Nombre de frames à garder en mémoire
        
        # === TRACKING ===
        self.track_history_detection = {}  # {id: [(x,y), ...]}
        self.track_history_segmentation = {}
        self.total_tracked_caps = set()  # IDs uniques vus (caps)
        self.total_tracked_bottles = set()  # IDs uniques vus (bottles)
        self.bottle_fill_status = {}  # {id: 0 ou 1} - Une fois remplie=1, reste à 1
        self.bottle_water_percentage = {}  # {id: pourcentage actuel}
        self.track_trail_length = 15  # Longueur des trails réduite pour performance
        
        # === ÉTAT ===
        self.paused = False
        self.recording = False
        self.video_writer = None
        self.show_help = True
        self.record_button_coords = (0, 0, 0, 0)  # Coordonnées du bouton d'enregistrement
        
        # === CACHE POUR OPTIMISATION ===
        self.cached_detection_frame = None
        self.cached_segmentation_frame = None
        self.cached_detection_data = (0, 0, None, False)
        self.cached_segmentation_data = (0, 0.0, None)
        
        # === DOSSIERS ===
        os.makedirs('results', exist_ok=True)
        os.makedirs('screenshots', exist_ok=True)
        os.makedirs('videos', exist_ok=True)
        
        # Charger les modèles
        self.load_models()
    
    def load_models(self):
        """Charge les deux modèles YOLOv8"""
        print("\n🔄 Chargement des modèles...")
        
        # Modèle 1: Détection
        if os.path.exists(self.model_detection_path):
            try:
                self.model_detection = YOLO(self.model_detection_path)
                print(f"✅ Modèle Détection chargé: {self.model_detection_path}")
            except Exception as e:
                print(f"❌ Erreur chargement détection: {e}")
        else:
            print(f"❌ Modèle détection introuvable: {self.model_detection_path}")
        
        # Modèle 2: Segmentation
        if os.path.exists(self.model_segmentation_path):
            try:
                self.model_segmentation = YOLO(self.model_segmentation_path)
                print(f"✅ Modèle Segmentation chargé: {self.model_segmentation_path}")
            except Exception as e:
                print(f"❌ Erreur chargement segmentation: {e}")
        else:
            print(f"❌ Modèle segmentation introuvable: {self.model_segmentation_path}")
        
        # Vérifier GPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Device: {device}")
        
        if self.model_detection is None or self.model_segmentation is None:
            print("\n⚠️ ATTENTION: Au moins un modèle n'a pas pu être chargé!")
            return False
        
        return True
    
    def get_color_by_id(self, track_id):
        """Génère une couleur unique basée sur l'ID de tracking"""
        np.random.seed(int(track_id))
        return tuple(np.random.randint(0, 255, 3).tolist())
    
    def draw_track_trail(self, frame, track_history, track_id, color):
        """Dessine la trajectoire d'un objet tracké"""
        if track_id in track_history:
            points = track_history[track_id]
            for i in range(1, len(points)):
                if points[i - 1] is None or points[i] is None:
                    continue
                thickness = int(np.sqrt(float(i + 1)) * 1.5)
                cv2.line(frame, points[i - 1], points[i], color, thickness)
    
    def process_detection(self, frame):
        """
        Applique le modèle de détection avec tracking (bounding boxes)
        
        Args:
            frame: Image OpenCV
            
        Returns:
            Image annotée, nombre d'objets actifs, total objets trackés, bool si bouteille détectée
        """
        if self.model_detection is None:
            return frame, 0, 0, False
        
        # Transposer si nécessaire
        processed_frame = frame.copy()
        if self.transpose_image:
            processed_frame = np.transpose(processed_frame, (1, 0, 2))
            processed_frame = np.ascontiguousarray(processed_frame)
        
        # Inférence avec TRACKING
        results = self.model_detection.track(processed_frame, 
                                             conf=self.confidence_detection, 
                                             iou=self.iou,
                                             persist=True,
                                             tracker="bytetrack.yaml",
                                             verbose=False,
                                             imgsz=self.img_size,
                                             half=False)
        
        # Dessiner les bounding boxes avec tracking
        annotated = processed_frame.copy()
        num_objects = 0  # Comptera uniquement les caps actifs
        bottle_detected = False  # Indicateur de présence de bouteille
        
        if results and len(results) > 0:
            result = results[0]
            
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                # Extraire les IDs de tracking si disponibles
                track_ids = None
                if hasattr(result.boxes, 'id') and result.boxes.id is not None:
                    track_ids = result.boxes.id.cpu().numpy().astype(int)
                
                for idx, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                    x1, y1, x2, y2 = map(int, box)
                    class_id = int(cls)
                    class_name = self.detection_classes.get(class_id, f"Class {class_id}")
                    
                    # Marquer si une bouteille est détectée
                    if class_name.lower() == 'bottle':
                        bottle_detected = True
                        continue  # Ne pas afficher les bouteilles, seulement les caps
                    
                    # Compter uniquement les caps
                    num_objects += 1
                    
                    # Récupérer l'ID de tracking
                    track_id = track_ids[idx] if track_ids is not None else None
                    
                    if track_id is not None:
                        # Ajouter à l'ensemble des IDs trackés
                        self.total_tracked_caps.add(track_id)
                        
                        # Mettre à jour l'historique de position
                        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        if track_id not in self.track_history_detection:
                            self.track_history_detection[track_id] = []
                        self.track_history_detection[track_id].append(center)
                        
                        # Limiter la longueur du trail
                        if len(self.track_history_detection[track_id]) > self.track_trail_length:
                            self.track_history_detection[track_id].pop(0)
                        
                        # Couleur unique par ID
                        color = self.get_color_by_id(track_id)
                        
                        # Dessiner le trail
                        self.draw_track_trail(annotated, self.track_history_detection, track_id, color)
                        
                        # Rectangle avec couleur unique
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                        
                        # Label avec ID
                        label = f"ID:{track_id} {class_name}: {conf:.2f}"
                    else:
                        color = self.detection_colors.get(class_id, (255, 255, 255))
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        label = f"{class_name}: {conf:.2f}"
                    
                    # Fond et texte du label
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(annotated, 
                                (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0] + 5, y1), 
                                color, -1)
                    cv2.putText(annotated, label, (x1 + 2, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Récupérer l'ID et la classe du premier cap visible (pour affichage du statut actuel)
        current_cap_id = None
        current_cap_has_cap = None  # True si "Avec Bouchon", False si "Sans Bouchon"
        
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                if hasattr(result.boxes, 'id') and result.boxes.id is not None:
                    classes_check = result.boxes.cls.cpu().numpy()
                    ids_check = result.boxes.id.cpu().numpy().astype(int)
                    
                    for cls, tid in zip(classes_check, ids_check):
                        class_name = self.detection_classes.get(int(cls), '')
                        # Ignorer les bouteilles
                        if class_name.lower() != 'bottle':
                            current_cap_id = tid
                            current_cap_has_cap = (class_name.lower() == 'avec bouchon')
                            break  # Prendre le premier cap
        
        return annotated, num_objects, len(self.total_tracked_caps), current_cap_has_cap, bottle_detected
    
    def process_segmentation(self, frame):
        """
        Applique le modèle de segmentation (masques)
        
        Args:
            frame: Image OpenCV
            
        Returns:
            Image annotée, nombre d'objets segmentés, pourcentage eau/bouteille, ID bouteille actuelle
        """
        if self.model_segmentation is None:
            return frame, 0, 0.0, None
        
        # Inférence avec TRACKING
        results = self.model_segmentation.track(frame, 
                                               conf=self.confidence_segmentation, 
                                               iou=self.iou,
                                               persist=True,
                                               tracker="bytetrack.yaml",
                                               verbose=False,
                                               imgsz=self.img_size,
                                               half=False)
        
        annotated = frame.copy()
        num_objects = 0
        
        # Variables pour calculer le pourcentage
        water_pixels = 0
        bottle_pixels = 0
        water_percentage = 0.0
        current_bottle_mask = None
        current_bottle_id = None
        
        if results and len(results) > 0:
            result = results[0]
            
            # Dessiner les masques de segmentation
            if hasattr(result, 'masks') and result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                num_objects = len(masks)
                
                # Extraire les IDs de tracking
                track_ids = None
                if hasattr(result.boxes, 'id') and result.boxes.id is not None:
                    track_ids = result.boxes.id.cpu().numpy().astype(int)
                
                for i, mask in enumerate(masks):
                    # Redimensionner le masque
                    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)
                    
                    # Classe et couleur
                    class_id = int(result.boxes.cls[i].cpu().numpy())
                    confidence = float(result.boxes.conf[i].cpu().numpy())
                    class_name = self.segmentation_classes.get(class_id, '')
                    
                    # ID de tracking
                    track_id = track_ids[i] if track_ids is not None else None
                    
                    # Si c'est une bouteille, tracker l'état de remplissage et la transparence
                    if track_id is not None and class_name.lower() == 'bottle':
                        self.total_tracked_bottles.add(track_id)
                        
                        # Initialiser le statut si nouveau
                        if track_id not in self.bottle_fill_status:
                            self.bottle_fill_status[track_id] = 0
                        
                        # Sauvegarder le masque et l'ID de la bouteille actuelle (pour affichage)
                        current_bottle_mask = mask_binary
                        current_bottle_id = track_id
                    
                    # Couleur unique par ID si tracking actif
                    if track_id is not None:
                        color = self.get_color_by_id(track_id)
                    else:
                        color = self.segmentation_colors.get(class_id, (255, 255, 255))
                    
                    # Compter les pixels par classe
                    pixel_count = np.sum(mask_binary)
                    if class_name.lower() == 'water':
                        water_pixels += pixel_count
                    elif class_name.lower() == 'bottle':
                        bottle_pixels += pixel_count
                    
                    # Overlay coloré semi-transparent
                    overlay = annotated.copy()
                    overlay[mask_binary == 1] = color
                    cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)
                    
                    # Contour du masque
                    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(annotated, contours, -1, color, 3)
                
                # Calculer le pourcentage eau/bouteille
                if bottle_pixels > 0:
                    water_percentage = (water_pixels / bottle_pixels) * 100
                
                # Mettre à jour le statut de remplissage pour les bouteilles trackées
                if hasattr(result.boxes, 'id') and result.boxes.id is not None:
                    track_ids_for_status = result.boxes.id.cpu().numpy().astype(int)
                    classes_for_status = result.boxes.cls.cpu().numpy()
                    
                    for tid, cls in zip(track_ids_for_status, classes_for_status):
                        class_name = self.segmentation_classes.get(int(cls), '')
                        if class_name.lower() == 'bottle':
                            # Si ce n'est pas encore dans le dictionnaire, l'initialiser
                            if tid not in self.bottle_fill_status:
                                self.bottle_fill_status[tid] = 0
                            
                            # Si le pourcentage est > 0, marquer comme remplie définitivement
                            if water_percentage > 0:
                                self.bottle_fill_status[tid] = 1
                            
                            # Sauvegarder le pourcentage actuel pour cette bouteille
                            self.bottle_water_percentage[tid] = water_percentage
            
            # Dessiner les bounding boxes avec labels
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                # Extraire les IDs pour les boxes
                track_ids_boxes = None
                if hasattr(result.boxes, 'id') and result.boxes.id is not None:
                    track_ids_boxes = result.boxes.id.cpu().numpy().astype(int)
                
                for idx, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                    x1, y1, x2, y2 = map(int, box)
                    class_id = int(cls)
                    class_name = self.segmentation_classes.get(class_id, f"Class {class_id}")
                    
                    # Récupérer l'ID et la couleur
                    track_id = track_ids_boxes[idx] if track_ids_boxes is not None else None
                    
                    if track_id is not None:
                        color = self.get_color_by_id(track_id)
                        
                        # Mettre à jour historique pour trail
                        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        if track_id not in self.track_history_segmentation:
                            self.track_history_segmentation[track_id] = []
                        self.track_history_segmentation[track_id].append(center)
                        if len(self.track_history_segmentation[track_id]) > self.track_trail_length:
                            self.track_history_segmentation[track_id].pop(0)
                        
                        # Dessiner trail
                        self.draw_track_trail(annotated, self.track_history_segmentation, track_id, color)
                        
                        # Rectangle avec ID
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                        label = f"ID:{track_id} {class_name}: {conf:.2f}"
                    else:
                        color = self.segmentation_colors.get(class_id, (255, 255, 255))
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        label = f"{class_name}: {conf:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    cv2.rectangle(annotated, 
                                (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0] + 5, y1), 
                                color, -1)
                    
                    cv2.putText(annotated, label, (x1 + 2, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated, num_objects, water_percentage, current_bottle_id
    
    def is_bottle_filled(self, track_id=None):
        """
        Détermine si la bouteille est remplie en utilisant le tracking
        Une fois qu'une bouteille a été détectée avec de l'eau (pourcentage > 0),
        elle reste marquée comme remplie même si le pourcentage retombe à 0
        
        Args:
            track_id: ID de la bouteille trackée (si disponible)
            
        Returns:
            1 si remplie (a déjà eu un pourcentage > 0)
            0 si vide (jamais eu de pourcentage > 0)
        """
        # Si on a un track_id, utiliser le statut tracké
        if track_id is not None and track_id in self.bottle_fill_status:
            return self.bottle_fill_status[track_id]
        
        # Sinon, utiliser l'historique global (comportement par défaut)
        if len(self.water_percentage_history) < 5:
            return 0  # Pas assez de données
        
        # Vérifier si tous les pourcentages sont à 0
        all_zero = all(p == 0.0 for p in self.water_percentage_history)
        
        if all_zero:
            return 0  # Bouteille vide
        else:
            return 1  # Bouteille remplie (pourcentage > 0)
    
    def draw_panel_info(self, frame, title, fps, conf, num_objects, water_percentage=None, current_bottle_id=None, current_cap_has_cap=None, is_left=True):
        """
        Dessine les informations sur un panneau
        
        Args:
            frame: Image à annoter
            title: Titre du panneau
            fps: FPS actuel
            conf: Niveau de confiance
            num_objects: Nombre d'objets détectés
            water_percentage: Pourcentage eau/bouteille (si disponible)
            current_bottle_id: ID de la bouteille actuelle
            current_cap_has_cap: Statut du bouchon
            is_left: True si panneau de gauche, False si droite
        """
        h, w = frame.shape[:2]
        
        # Position du texte
        x = 10 if is_left else 10
        
        # Fond semi-transparent pour le titre
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 35), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Titre
        cv2.putText(frame, title, (x, 25), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
        
        # Informations - calculer les infos d'abord pour déterminer la hauteur
        infos = [
            f"FPS: {fps:.1f}",
            f"Conf: {conf:.2f}",
            f"Active: {num_objects}"
        ]
        
        # Ajouter stats de tracking selon le panneau
        if water_percentage is None:
            # Panneau gauche (détection caps) - Afficher uniquement le statut avec bouchon
            # Ajouter le statut "avec bouchon" du cap actuel
            if current_cap_has_cap is not None:
                cap_status = "OUI" if current_cap_has_cap else "NON"
                infos.append(f"Bouteille actuelle")
                infos.append(f"avec bouchon: {cap_status}")
        else:
            # Panneau droite (segmentation) - Afficher Water, statut remplie et transparence
            infos.append(f"Water: {water_percentage:.1f}%")
            
            # Ajouter le statut de la bouteille ACTUELLE dans la caméra
            if current_bottle_id is not None:
                if current_bottle_id in self.bottle_fill_status:
                    current_status = "OUI" if self.bottle_fill_status[current_bottle_id] == 1 else "NON"
                    infos.append(f"Bouteille actuelle")
                    infos.append(f"est remplie: {current_status}")
        
        # Ajuster la hauteur selon le nombre de lignes
        line_height = 22
        info_height = 68 + len(infos) * line_height
        
        # Ajuster la largeur selon le contenu
        info_width = 280 if water_percentage is not None else 180
        
        # Dessiner le rectangle
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (x, 45), (x + info_width, info_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)
        
        for i, info in enumerate(infos):
            cv2.putText(frame, info, (x + 5, 68 + i * 22), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def draw_combined_interface(self, combined_frame, num_det, num_seg):
        """
        Dessine l'interface globale sur la frame combinée
        
        Args:
            combined_frame: Frame avec les deux modèles côte à côte
            num_det: Nombre d'objets détectés (modèle 1)
            num_seg: Nombre d'objets segmentés (modèle 2)
        """
        h, w = combined_frame.shape[:2]
        
        # === BARRE DU HAUT ===
        overlay = combined_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.8, combined_frame, 0.2, 0, combined_frame)
        
        # Titre principal centré
        title = "UNIFIED BOTTLE DETECTION - DUAL MODEL DISPLAY"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
        title_x = (w - title_size[0]) // 2
        cv2.putText(combined_frame, title, (title_x, 28), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
        
        # === LIGNE DE SÉPARATION VERTICALE ===
        mid_x = w // 2
        cv2.line(combined_frame, (mid_x, 40), (mid_x, h - 80), (255, 255, 255), 2)
        
        # === BARRE DU BAS (Contrôles) ===
        if self.show_help:
            overlay_bottom = combined_frame.copy()
            cv2.rectangle(overlay_bottom, (0, h - 80), (w, h), (30, 30, 30), -1)
            cv2.addWeighted(overlay_bottom, 0.8, combined_frame, 0.2, 0, combined_frame)
            
            controls = [
                "[SPACE] Pause/Resume",
                "[S] Screenshot",
                "[R] Start/Stop Recording",
                "[T] Transpose Image",
                "[H] Hide/Show Help",
                "[+/-] Adjust Confidence",
                "[Q/ESC] Quit"
            ]
            
            # Diviser les contrôles en 2 lignes
            line1 = " | ".join(controls[:4])
            line2 = " | ".join(controls[4:])
            
            cv2.putText(combined_frame, line1, (20, h - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(combined_frame, line2, (20, h - 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            # Afficher juste un hint
            cv2.putText(combined_frame, "[H] Show Help", (20, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # === INDICATEUR DE PAUSE ===
        if self.paused:
            pause_text = "|| PAUSED ||"
            pause_size = cv2.getTextSize(pause_text, cv2.FONT_HERSHEY_DUPLEX, 1.5, 3)[0]
            pause_x = (w - pause_size[0]) // 2
            
            # Fond
            cv2.rectangle(combined_frame, 
                         (pause_x - 10, 50), 
                         (pause_x + pause_size[0] + 10, 90), 
                         (0, 0, 0), -1)
            cv2.rectangle(combined_frame, 
                         (pause_x - 10, 50), 
                         (pause_x + pause_size[0] + 10, 90), 
                         (0, 0, 255), 3)
            
            cv2.putText(combined_frame, pause_text, (pause_x, 80), 
                       cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)
        
        # === BOUTON D'ENREGISTREMENT (cliquable) ===
        button_x = w - 180
        button_y = 50
        button_w = 160
        button_h = 40
        
        # Dessiner le bouton
        if self.recording:
            # Bouton rouge "STOP RECORDING"
            cv2.rectangle(combined_frame, (button_x, button_y), 
                         (button_x + button_w, button_y + button_h), (0, 0, 200), -1)
            cv2.rectangle(combined_frame, (button_x, button_y), 
                         (button_x + button_w, button_y + button_h), (0, 0, 255), 2)
            
            # Cercle rouge clignotant
            if int(time.time() * 2) % 2 == 0:
                cv2.circle(combined_frame, (button_x + 20, button_y + 20), 8, (0, 0, 255), -1)
            
            cv2.putText(combined_frame, "STOP REC", (button_x + 35, button_y + 27), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            # Bouton vert "START RECORDING"
            cv2.rectangle(combined_frame, (button_x, button_y), 
                         (button_x + button_w, button_y + button_h), (0, 100, 0), -1)
            cv2.rectangle(combined_frame, (button_x, button_y), 
                         (button_x + button_w, button_y + button_h), (0, 200, 0), 2)
            
            cv2.circle(combined_frame, (button_x + 20, button_y + 20), 8, (255, 255, 255), -1)
            
            cv2.putText(combined_frame, "START REC", (button_x + 35, button_y + 27), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Stocker les coordonnées du bouton pour la détection de clic
        self.record_button_coords = (button_x, button_y, button_x + button_w, button_y + button_h)
    
    def mouse_callback(self, event, x, y, flags, param):
        """
        Callback pour gérer les clics de souris sur l'interface
        
        Args:
            event: Type d'événement souris
            x, y: Coordonnées du clic
            flags: Flags additionnels
            param: Paramètres additionnels
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Vérifier si le clic est sur le bouton d'enregistrement
            x1, y1, x2, y2 = self.record_button_coords
            if x1 <= x <= x2 and y1 <= y <= y2:
                # Toggle enregistrement
                if not self.recording:
                    # Démarrer l'enregistrement
                    # On récupère les dimensions de la frame courante
                    if hasattr(self, 'current_frame_size'):
                        w, h = self.current_frame_size
                        self.start_recording(w, h)
                else:
                    # Arrêter l'enregistrement
                    self.stop_recording()
    
    def calculate_fps(self):
        """Calcule le FPS actuel"""
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.fps = self.frame_count / elapsed
    
    def save_screenshot(self, frame):
        """Sauvegarde une capture d'écran"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshots/screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot sauvegardé: {filename}")
    
    def start_recording(self, width, height):
        """Démarre l'enregistrement vidéo"""
        if not self.recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"videos/recording_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
            self.recording = True
            print(f"🔴 Enregistrement démarré: {filename}")
    
    def stop_recording(self):
        """Arrête l'enregistrement vidéo"""
        if self.recording and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.recording = False
            print("⏹️ Enregistrement arrêté")
    
    def webcam_mode(self):
        """Mode webcam avec affichage simultané des 2 modèles"""
        print("\n🎥 Démarrage du mode WEBCAM...")
        print("🔍 Recherche de la webcam...")
        
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
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("✅ Webcam configurée (640x480)")
        print("\n🚀 Lancement de l'interface unifiée...")
        print("📺 Les deux modèles s'afficheront côte à côte")
        
        window_name = 'Unified Bottle Detection - Dual Model'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1400, 600)
        
        # Ajouter le callback pour les clics de souris
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        # Reset statistiques
        self.frame_count = 0
        self.start_time = time.time()
        
        while True:
            if not self.paused:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Erreur lecture webcam")
                    break
                
                # Incrémenter le compteur de frames
                self.frame_skip_counter += 1
                
                # Traiter le modèle de détection tous les N frames
                bottle_detected = False
                if self.frame_skip_counter % self.process_every_n_frames == 0:
                    frame_left, num_det, total_caps, current_cap_has_cap, bottle_detected = self.process_detection(frame.copy())
                    self.cached_detection_frame = frame_left
                    self.cached_detection_data = (num_det, total_caps, current_cap_has_cap, bottle_detected)
                else:
                    frame_left = self.cached_detection_frame if self.cached_detection_frame is not None else frame.copy()
                    num_det, total_caps, current_cap_has_cap, bottle_detected = self.cached_detection_data
                
                # Traiter le modèle de segmentation UNIQUEMENT si une bouteille a été détectée
                if bottle_detected:
                    if (self.frame_skip_counter + 1) % self.process_every_n_frames == 0:
                        frame_right, num_seg, water_pct, current_bottle_id = self.process_segmentation(frame.copy())
                        self.cached_segmentation_frame = frame_right
                        self.cached_segmentation_data = (num_seg, water_pct, current_bottle_id)
                    else:
                        frame_right = self.cached_segmentation_frame if self.cached_segmentation_frame is not None else frame.copy()
                        num_seg, water_pct, current_bottle_id = self.cached_segmentation_data
                else:
                    # Pas de bouteille détectée : afficher la frame originale avec un message
                    frame_right = frame.copy()
                    num_seg, water_pct, current_bottle_id = 0, 0.0, None
                    # Ajouter un message "En attente de bouteille..."
                    h, w = frame_right.shape[:2]
                    overlay = frame_right.copy()
                    cv2.rectangle(overlay, (w//2 - 200, h//2 - 40), (w//2 + 200, h//2 + 40), (50, 50, 50), -1)
                    cv2.addWeighted(overlay, 0.8, frame_right, 0.2, 0, frame_right)
                    cv2.putText(frame_right, "En attente de bouteille...", (w//2 - 180, h//2 + 10), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
                
                # Mettre à jour l'historique des pourcentages
                self.water_percentage_history.append(water_pct)
                if len(self.water_percentage_history) > self.history_size:
                    self.water_percentage_history.pop(0)  # Garder seulement les N dernières valeurs
                
                # S'assurer que les deux frames ont la même taille
                # (Important si transpose est activé sur le modèle de détection)
                if frame_left.shape != frame_right.shape:
                    # Redimensionner frame_left pour correspondre à frame_right
                    frame_left = cv2.resize(frame_left, (frame_right.shape[1], frame_right.shape[0]))
                
                # Calculer FPS
                self.calculate_fps()
                
                # Ajouter les infos sur chaque panneau
                self.draw_panel_info(frame_left, "DETECTION (Bottle+Cap)", 
                                    self.fps, self.confidence_detection, num_det, None, None, current_cap_has_cap, is_left=True)
                self.draw_panel_info(frame_right, "SEGMENTATION (Bottle+Water)", 
                                    self.fps, self.confidence_segmentation, num_seg, water_pct, current_bottle_id, None, is_left=False)
                
                # Combiner les deux frames côte à côte
                combined = np.hstack([frame_left, frame_right])
                
                # Sauvegarder les dimensions pour le callback souris
                h_combined, w_combined = combined.shape[:2]
                self.current_frame_size = (w_combined, h_combined)
                
                # Ajouter l'interface globale
                self.draw_combined_interface(combined, num_det, num_seg)
                
                # Enregistrer si actif
                if self.recording and self.video_writer is not None:
                    self.video_writer.write(combined)
                
                current_frame = combined
            
            # Afficher
            cv2.imshow(window_name, current_frame)
            
            # Gestion clavier (waitKey plus court pour meilleure réactivité)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27 or key == ord('q'):  # ESC ou Q
                break
            elif key == ord(' '):  # SPACE - Pause
                self.paused = not self.paused
                print(f"{'⏸️ PAUSE' if self.paused else '▶️ REPRISE'}")
            elif key == ord('s'):  # S - Screenshot
                self.save_screenshot(current_frame)
            elif key == ord('r'):  # R - Recording
                if not self.recording:
                    h, w = current_frame.shape[:2]
                    self.start_recording(w, h)
                else:
                    self.stop_recording()
            elif key == ord('t'):  # T - Transpose
                self.transpose_image = not self.transpose_image
                print(f"🔄 Transpose: {'ON' if self.transpose_image else 'OFF'}")
            elif key == ord('h'):  # H - Help
                self.show_help = not self.show_help
            elif key == ord('+') or key == ord('='):  # + - Augmenter confiance
                self.confidence_detection = min(0.95, self.confidence_detection + 0.05)
                self.confidence_segmentation = min(0.95, self.confidence_segmentation + 0.05)
                print(f"⬆️ Confidence: Det={self.confidence_detection:.2f}, Seg={self.confidence_segmentation:.2f}")
            elif key == ord('-') or key == ord('_'):  # - - Diminuer confiance
                self.confidence_detection = max(0.1, self.confidence_detection - 0.05)
                self.confidence_segmentation = max(0.1, self.confidence_segmentation - 0.05)
                print(f"⬇️ Confidence: Det={self.confidence_detection:.2f}, Seg={self.confidence_segmentation:.2f}")
        
        # Nettoyage
        if self.recording:
            self.stop_recording()
        
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Interface fermée")
    
    def single_image_mode(self):
        """Mode image unique avec les 2 modèles"""
        print("\n🖼️ Mode IMAGE UNIQUE")
        print("📁 Entrez le chemin de l'image:")
        
        image_path = input("Chemin: ").strip().strip('"').strip("'")
        
        if not os.path.exists(image_path):
            print(f"❌ Image introuvable: {image_path}")
            return
        
        # Charger image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ Impossible de lire l'image: {image_path}")
            return
        
        print(f"✅ Image chargée: {frame.shape}")
        print("🔄 Application des modèles...")
        
        # Appliquer les deux modèles
        frame_left, num_det, total_caps, current_cap_has_cap, bottle_detected = self.process_detection(frame.copy())
        
        # Appliquer segmentation uniquement si bouteille détectée
        if bottle_detected:
            frame_right, num_seg, water_pct, current_bottle_id = self.process_segmentation(frame.copy())
        else:
            frame_right = frame.copy()
            num_seg, water_pct, current_bottle_id = 0, 0.0, None
            h, w = frame_right.shape[:2]
            cv2.putText(frame_right, "Aucune bouteille detectee", (w//2 - 150, h//2), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
        
        # Ajouter les infos
        self.draw_panel_info(frame_left, "DETECTION (Bottle+Cap)", 
                            0, self.confidence_detection, num_det, None, None, current_cap_has_cap, is_left=True)
        self.draw_panel_info(frame_right, "SEGMENTATION (Bottle+Water)", 
                            0, self.confidence_segmentation, num_seg, water_pct, current_bottle_id, None, is_left=False)
        
        # Combiner
        combined = np.hstack([frame_left, frame_right])
        self.draw_combined_interface(combined, num_det, num_seg)
        
        # Afficher
        window_name = f'Unified Detection - {os.path.basename(image_path)}'
        cv2.imshow(window_name, combined)
        
        print("✅ Résultat affiché")
        print("💾 Appuyez sur 'S' pour sauvegarder, ou n'importe quelle touche pour fermer")
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"results/result_{timestamp}_{os.path.basename(image_path)}"
            cv2.imwrite(output_path, combined)
            print(f"💾 Résultat sauvegardé: {output_path}")
        
        cv2.destroyAllWindows()
    
    def folder_mode(self):
        """Mode dossier avec les 2 modèles"""
        print("\n📁 Mode DOSSIER MULTIPLE")
        print("📂 Entrez le chemin du dossier:")
        
        folder_path = input("Chemin: ").strip().strip('"').strip("'")
        
        if not os.path.exists(folder_path):
            print(f"❌ Dossier introuvable: {folder_path}")
            return
        
        # Liste des images
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in extensions:
            image_files.extend(list(Path(folder_path).glob(f'*{ext}')))
            image_files.extend(list(Path(folder_path).glob(f'*{ext.upper()}')))
        
        if len(image_files) == 0:
            print(f"❌ Aucune image trouvée dans: {folder_path}")
            return
        
        print(f"✅ {len(image_files)} images trouvées")
        print("🔄 Traitement en cours...")
        
        # Créer sous-dossier pour les résultats
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = f"results/batch_{timestamp}"
        os.makedirs(output_folder, exist_ok=True)
        
        for i, image_path in enumerate(image_files, 1):
            print(f"   [{i}/{len(image_files)}] {image_path.name}...", end='')
            
            # Charger image
            frame = cv2.imread(str(image_path))
            if frame is None:
                print(" ❌ Erreur lecture")
                continue
            
            # Appliquer les modèles
            frame_left, num_det, total_caps, current_cap_has_cap, bottle_detected = self.process_detection(frame.copy())
            
            # Appliquer segmentation uniquement si bouteille détectée
            if bottle_detected:
                frame_right, num_seg, water_pct, current_bottle_id = self.process_segmentation(frame.copy())
            else:
                frame_right = frame.copy()
                num_seg, water_pct, current_bottle_id = 0, 0.0, None
            
            # Ajouter infos
            self.draw_panel_info(frame_left, "DETECTION", 0, self.confidence_detection, num_det, None, None, current_cap_has_cap)
            self.draw_panel_info(frame_right, "SEGMENTATION", 0, self.confidence_segmentation, num_seg, water_pct, current_bottle_id, None)
            
            # Combiner
            combined = np.hstack([frame_left, frame_right])
            self.draw_combined_interface(combined, num_det, num_seg)
            
            # Sauvegarder
            output_path = os.path.join(output_folder, f"unified_{image_path.name}")
            cv2.imwrite(output_path, combined)
            
            print(f" ✅ Det:{num_det} Seg:{num_seg}")
        
        print(f"\n✅ Traitement terminé!")
        print(f"📁 Résultats sauvegardés dans: {output_folder}")


def main():
    """Point d'entrée principal"""
    print("\n" + "=" * 70)
    print("🍶 UNIFIED BOTTLE DETECTION SYSTEM")
    print("   Affichage simultané: Détection (Bottle+Cap) | Segmentation (Bottle+Water)")
    print("=" * 70)
    
    # Initialiser
    detector = UnifiedBottleDetection()
    
    if detector.model_detection is None or detector.model_segmentation is None:
        print("\n❌ Les modèles n'ont pas pu être chargés correctement.")
        print("💡 Vérifiez les chemins vers les fichiers .pt")
        input("\nAppuyez sur Entrée pour quitter...")
        return
    
    # Menu principal
    while True:
        print("\n" + "=" * 70)
        print("📺 MENU PRINCIPAL")
        print("=" * 70)
        print("1. Mode WEBCAM (temps réel)")
        print("2. Mode IMAGE UNIQUE")
        print("3. Mode DOSSIER MULTIPLE")
        print("4. Quitter")
        print("=" * 70)
        
        choice = input("\nVotre choix (1-4): ").strip()
        
        if choice == '1':
            detector.webcam_mode()
        elif choice == '2':
            detector.single_image_mode()
        elif choice == '3':
            detector.folder_mode()
        elif choice == '4':
            print("\n👋 Au revoir!")
            break
        else:
            print("❌ Choix invalide!")
    
    print("\n✅ Programme terminé")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Programme interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        input("\nAppuyez sur Entrée pour quitter...")
