# Comparaison YOLOv8 pre-entraine contre YOLOv8 fine-tune

Split evalue : `test`
Modele de reference : `models/yolov8n.pt`
Modele fine-tune : `models/finetuned/yolov8n_benin.pt`

| Metrique | Pre-entraine | Fine-tune | Ecart |
|---|---|---|---|
| mAP@0.5 | 0.4380 | 0.8252 | +88.4 % |
| mAP@0.5:0.95 | 0.2959 | 0.6443 | +117.8 % |
| Precision | 0.4959 | 0.8327 | +67.9 % |
| Recall | 0.4172 | 0.7138 | +71.1 % |
| F1-score | 0.4532 | 0.7687 | +69.6 % |
| Debit (FPS) | 37.2 | 43.2 | +16.2 % |
| Rappel objets petits (804 objets) | 0.1219 | 0.6940 | +469.4 % |
| Rappel objets moyens (1415 objets) | 0.3731 | 0.8792 | +135.6 % |
| Rappel objets grands (332 objets) | 0.7922 | 0.9518 | +20.2 % |

Le rappel par taille d'objet suit la convention COCO : petit sous 32x32
pixels, moyen sous 96x96, grand au-dela. C'est la ligne des objets petits
qui repond directement a la problematique des vehicules eloignes.
