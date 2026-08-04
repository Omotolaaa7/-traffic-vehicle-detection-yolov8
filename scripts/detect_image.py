"""Detection de vehicules sur une image ou un dossier d'images.

Sert a produire les visuels qualitatifs du rapport : la meme image passee dans
le modele pre-entraine puis dans le modele fine-tune, cote a cote.

Usage :
    python scripts/detect_image.py
    python scripts/detect_image.py data/raw/embouteillages2.jpg
    python scripts/detect_image.py data/raw/ --model models/finetuned/yolov8n_benin.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

RACINE = Path(__file__).resolve().parent.parent
MODELE_DEFAUT = RACINE / "models" / "yolov8n.pt"
ENTREE_DEFAUT = RACINE / "data" / "raw" / "embouteillages1.jpg"
SORTIE_DEFAUT = RACINE / "data" / "outputs"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
# Classes COCO retenues : car, motorcycle, bus, truck.
COCO_VEHICULES = [2, 3, 5, 7]


def resoudre(chemin: Path) -> Path:
    return chemin if chemin.is_absolute() else RACINE / chemin


def lister_images(entree: Path) -> list[Path]:
    if entree.is_dir():
        return [p for p in sorted(entree.iterdir()) if p.suffix.lower() in EXTENSIONS]
    return [entree]


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("entree", nargs="?", type=Path, default=ENTREE_DEFAUT,
                         help="image ou dossier d'images")
    parseur.add_argument("--model", type=Path, default=MODELE_DEFAUT)
    parseur.add_argument("--out", type=Path, default=SORTIE_DEFAUT)
    parseur.add_argument("--conf", type=float, default=0.25)
    args = parseur.parse_args()

    entree, modele_path, sortie = resoudre(args.entree), resoudre(args.model), resoudre(args.out)
    if not entree.exists():
        raise SystemExit(f"Entree introuvable : {entree}")
    if not modele_path.exists():
        raise SystemExit(f"Modele introuvable : {modele_path}")

    images = lister_images(entree)
    if not images:
        raise SystemExit(f"Aucune image dans {entree}")

    sortie.mkdir(parents=True, exist_ok=True)
    modele = YOLO(str(modele_path))
    # Le modele pre-entraine connait 80 classes : on ne garde que les vehicules.
    # Le modele fine-tune n'en connait que 4, toutes pertinentes.
    classes = COCO_VEHICULES if len(modele.names) >= 80 else None

    for image in images:
        resultat = modele.predict(str(image), conf=args.conf, classes=classes, verbose=False)[0]
        destination = sortie / f"{image.stem}_{modele_path.stem}.jpg"
        resultat.save(filename=str(destination))
        print(f"{image.name:<40} {len(resultat.boxes):>3} vehicule(s) -> {destination.name}")


if __name__ == "__main__":
    main()
