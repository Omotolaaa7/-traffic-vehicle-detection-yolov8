"""Etape 5 de docs/PROCEDURE.md : fine-tuning de YOLOv8.

Tous les hyperparametres viennent de configs/entrainement.yaml. Aucun n'est
code en dur ici, afin qu'un run soit entierement decrit par son fichier de
configuration et donc reproductible.

Usage :
    python scripts/train_yolo.py
    python scripts/train_yolo.py --config configs/entrainement.yaml --epochs 100
"""

import argparse
import shutil
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

RACINE = Path(__file__).resolve().parent.parent
CONFIG_DEFAUT = RACINE / "configs" / "entrainement.yaml"
DESTINATION_POIDS = RACINE / "models" / "finetuned"


def charger_config(chemin: Path) -> dict:
    if not chemin.exists():
        sys.exit(f"Configuration introuvable : {chemin}")
    return yaml.safe_load(chemin.read_text(encoding="utf-8"))


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--config", type=Path, default=CONFIG_DEFAUT)
    parseur.add_argument("--epochs", type=int, help="surcharge la valeur du fichier de config")
    parseur.add_argument("--device", help="surcharge le device (cpu, 0, ...)")
    args = parseur.parse_args()

    config = charger_config(args.config)
    augmentation = config.get("augmentation", {})

    donnees = RACINE / config["donnees"]
    if not donnees.exists():
        sys.exit(
            f"{donnees} introuvable.\n"
            "Lancez d'abord : python scripts/import_dataset.py"
        )

    modele_base = RACINE / config["modele_base"]
    print(f"Modele de depart : {modele_base}")
    print(f"Jeu de donnees   : {donnees}")

    modele = YOLO(str(modele_base))
    resultats = modele.train(
        data=str(donnees),
        epochs=args.epochs or config["epochs"],
        imgsz=config["taille_image"],
        batch=config["batch"],
        patience=config["patience"],
        device=args.device if args.device is not None else config["device"],
        workers=config["workers"],
        seed=config["seed"],
        project=str(RACINE / config["projet"]),
        name=config["nom_run"],
        exist_ok=False,
        **augmentation,
    )

    # Ultralytics ecrit les poids dans le dossier du run ; on copie le meilleur
    # dans models/finetuned/ pour que evaluate.py et compare_models.py aient un
    # chemin stable a viser.
    meilleur = Path(resultats.save_dir) / "weights" / "best.pt"
    if meilleur.exists():
        DESTINATION_POIDS.mkdir(parents=True, exist_ok=True)
        cible = DESTINATION_POIDS / f"{config['nom_run']}.pt"
        shutil.copy2(meilleur, cible)
        print(f"\nMeilleurs poids copies dans {cible}")
        print("Etape suivante :")
        print(f"  python scripts/compare_models.py --finetuned {cible.relative_to(RACINE)}")
    else:
        print(f"\nAttention : {meilleur} introuvable, poids non copies.")


if __name__ == "__main__":
    main()
