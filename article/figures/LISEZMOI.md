# Figures de l'article

## Déjà en place (générées localement le 6 août 2026)

- `scene1_base.jpg` et `scene1_finetune.jpg` : détections des deux
  modèles sur `val_000556.jpg` du jeu de test (seuil 0,25, boîtes sans
  étiquettes pour la lisibilité). Le modèle pré-entraîné y détecte
  4 véhicules, le fine-tuné 44. Visages floutés à la source dans
  BMD-45 ; la plaque de la voiture blanche (seule partiellement
  lisible) a été floutée avec un flou gaussien.

## Reste à déposer

- `courbes_entrainement.pdf` : les courbes d'entraînement (pertes et
  mAP par epoch). Les données sont dans le dossier `runs/` de la
  session Colab d'entraînement (fichiers `results.png` et
  `results.csv`). Une fois le fichier déposé, décommenter la ligne
  `\includegraphics` correspondante dans `sections/resultats.tex` et
  supprimer le bloc `\todo`. Si le PNG est utilisé tel quel, adapter
  l'extension dans `\includegraphics`.
