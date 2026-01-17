# 🍶 Unified Bottle Detection System

Système unifié de détection et segmentation de bouteilles utilisant deux modèles YOLOv8 pour l'analyse en temps réel.

## 📋 Description

Ce projet combine deux modèles de deep learning pour une analyse complète des bouteilles :

### Modèle 1 : Détection (Bounding Boxes)
- **Détection de bouteilles** avec ou sans bouchon
- **Classes détectées** : 
  - Bouteille
  - Avec Bouchon
  - Sans Bouchon
- **Tracking** : Suivi des objets avec trajectoires

### Modèle 2 : Segmentation 
- **Segmentation précise** des bouteilles et du niveau d'eau
- **Classes segmentées** :
  - Bouteille
  - Eau
- **Analyse du remplissage** : Calcul du pourcentage d'eau

## ✨ Fonctionnalités

- ✅ Affichage simultané des deux modèles (côte à côte)
- ✅ Détection et tracking en temps réel
- ✅ Calcul automatique du pourcentage de remplissage
- ✅ Suivi des trajectoires des objets
- ✅ Support GPU (CUDA) pour performances optimales
- ✅ Enregistrement vidéo
- ✅ Captures d'écran
- ✅ Interface interactive avec contrôles clavier/souris
- ✅ Statistiques en temps réel (FPS, compteurs)

## 🚀 Installation

### Prérequis
```bash
Python 3.8+
CUDA (optionnel, pour accélération GPU)
```

### Installation des dépendances
```bash
pip install ultralytics opencv-python torch numpy
```

## 📁 Structure du Projet

```
.
├── unified_bottle_detection.py       # Script principal
├── Bottle-Bottle-Cap-Detection-System-main/
│   └── best.pt                       # Modèle de détection
├── remplie/remplie/3 segmentations.v4i.yolov8/
│   └── runs/segment/bottle_final_quality/weights/
│       └── best.pt                   # Modèle de segmentation
├── results/                          # Dossier pour les résultats
├── screenshots/                      # Captures d'écran
└── videos/                          # Vidéos enregistrées
```

## 🎮 Utilisation

### Lancer le programme
```bash
python unified_bottle_detection.py
```

### Contrôles Clavier

| Touche | Action |
|--------|--------|
| `h` / `H` | Afficher/masquer l'aide |
| `p` / `ESPACE` | Pause/Reprendre |
| `s` | Capturer une screenshot |
| `r` | Démarrer/arrêter l'enregistrement vidéo |
| `t` | Activer/désactiver la transposition |
| `+` / `-` | Ajuster la confiance (détection) |
| `[` / `]` | Ajuster la confiance (segmentation) |
| `q` / `ESC` | Quitter |

### Contrôles Souris
- **Clic gauche** sur le bouton REC : Démarrer/arrêter l'enregistrement

## 📊 Paramètres

- **Confiance détection** : 0.5 (ajustable avec +/-)
- **Confiance segmentation** : 0.3 (ajustable avec [/])
- **IOU** : 0.5
- **Taille image** : 640x640
- **Device** : GPU (CUDA) si disponible, sinon CPU

## 🎯 Statistiques Affichées

### Modèle Détection
- Nombre d'objets actifs dans la frame
- Total d'objets trackés depuis le début
- Pourcentage de bouteilles avec bouchon

### Modèle Segmentation
- Nombre d'objets actifs
- Pourcentage moyen de remplissage des bouteilles
- État de remplissage par bouteille

## 💾 Sorties

- **Screenshots** : Sauvegardées dans `screenshots/` au format PNG
- **Vidéos** : Enregistrées dans `videos/` au format MP4 (codec H264)
- **Format** : Double affichage (détection + segmentation côte à côte)

## 🔧 Configuration GPU

Le système détecte automatiquement la disponibilité du GPU CUDA. Pour forcer le CPU :
```python
self.device = 'cpu'
```

## 📈 Performances

- **FPS** : Affichage en temps réel
- **Optimisations** :
  - Traitement GPU
  - Cache des frames
  - Skip frames configurables
  - Trails de trajectoire optimisés

## 🐛 Dépannage

### Le modèle ne se charge pas
Vérifiez que les chemins des modèles sont corrects dans le code :
```python
self.model_detection_path = 'Bottle-Bottle-Cap-Detection-System-main/best.pt'
self.model_segmentation_path = 'remplie/remplie/3 segmentations.v4i.yolov8/runs/segment/bottle_final_quality/weights/best.pt'
```

### Problèmes de caméra
Changez l'index de la caméra dans la fonction `main()` :
```python
cap = cv2.VideoCapture(0)  # Essayez 1, 2, etc.
```

### Performance lente
- Vérifiez que CUDA est bien installé
- Réduisez la taille de l'image (`img_size`)
- Augmentez `process_every_n_frames`

## 📝 Auteur

HATIM ABDESSAMAD

## 📄 Licence

Ce projet est destiné à un usage éducatif et de recherche.

## 🙏 Remerciements

- YOLOv8 par Ultralytics
- OpenCV pour le traitement d'image
- PyTorch pour le deep learning
