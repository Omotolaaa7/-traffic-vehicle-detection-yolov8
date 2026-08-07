# Figures de l'article

## `scene1_base.jpg` et `scene1_finetune.jpg`

Générées localement le 6 août 2026. Détections des deux modèles sur
`val_000556.jpg` du jeu de test (seuil 0,25, boîtes sans étiquettes pour
la lisibilité). Le modèle pré-entraîné y détecte 4 véhicules, le
fine-tuné 44. Visages floutés à la source dans BMD-45 ; la plaque de la
voiture blanche (seule partiellement lisible) a été floutée avec un flou
gaussien.

## `courbes_entrainement.pdf`

Générée le 7 août 2026. Ne pas éditer à la main : c'est une sortie de
script, régénérable par

```bash
python scripts/tracer_courbes.py
```

Le script lit `results/entrainements/yolov8n_benin/results.csv`, produit
par le fine-tuning. Ce CSV n'est pas versionné (`.gitignore`) : si le
dossier est absent d'une copie du dépôt, récupérez-le depuis la
sauvegarde Drive du notebook Colab (`Projet1_AMA/results/`), comme
indiqué dans `docs/PROCEDURE.md`.

Le `results.png` produit par Ultralytics dans le même dossier trace les
mêmes données, mais nous le retraçons en PDF vectoriel : sa police
devient illisible une fois l'image réduite à la largeur de la page, et
son rendu matriciel pixellise à l'impression.
