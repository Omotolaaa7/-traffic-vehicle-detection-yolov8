"""Etape 4 de docs/PROCEDURE.md : evaluation d'un modele de detection.

Calcule les metriques annoncees dans le rapport technique :
    mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1-score, temps d'inference (FPS)

Et surtout la mesure propre au sujet : le **rappel par taille d'objet**.
La problematique porte sur les vehicules eloignes et de petite taille ; une
moyenne globale masque precisement cet effet. Les seuils d'aire suivent la
convention COCO (petit < 32x32 px, moyen < 96x96 px, grand au-dela).

Le script gere les deux indexations de classes :
  - modele pre-entraine COCO  -> 80 classes, vehicules aux indices 2, 3, 5, 7
  - modele fine-tune          -> 4 classes, indices 0 a 3
Pour que la comparaison porte sur le meme jeu de test, les labels sont
reindexes a la volee vers l'espace de classes du modele evalue.

Usage :
    python scripts/evaluate.py --model models/yolov8n.pt
    python scripts/evaluate.py --model models/finetuned/yolov8n_benin.pt --split test
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

RACINE = Path(__file__).resolve().parent.parent
CONFIG_DATASET = RACINE / "configs" / "dataset.yaml"
DOSSIER_RESULTATS = RACINE / "results"

# Correspondance entre les classes locales et les classes COCO du modele
# pre-entraine. Cette table est la condition d'une comparaison honnete.
LOCAL_VERS_COCO = {0: 2, 1: 3, 2: 5, 3: 7}  # voiture, moto, bus, camion
COCO_VEHICULES = sorted(LOCAL_VERS_COCO.values())

SEUIL_CONFIANCE = 0.25
SEUIL_IOU = 0.5

# Seuils d'aire de la convention COCO, exprimes en pixels.
BORNES_TAILLE = [
    ("petit", 0, 32 * 32),
    ("moyen", 32 * 32, 96 * 96),
    ("grand", 96 * 96, float("inf")),
]

# Resolution a laquelle les aires sont ramenees avant d'etre classees.
#
# Les seuils COCO supposent des images d'environ 640 px. Nos photos font
# plusieurs milliers de pixels de large : appliquees telles quelles, elles
# rangeraient la totalite des vehicules dans la categorie "grand" et
# l'analyse ne mesurerait plus rien. On rapporte donc chaque aire a la
# resolution d'entree du reseau, qui est la taille a laquelle le detecteur
# voit reellement l'objet.
TAILLE_REFERENCE = 640


# --------------------------------------------------------------------------
# Preparation d'une vue du jeu de test dans l'espace de classes du modele
# --------------------------------------------------------------------------

def modele_est_coco(modele: YOLO) -> bool:
    """Un modele pre-entraine COCO expose 80 classes ; le notre en expose 4."""
    return len(modele.names) >= 80


def construire_vue_coco(dossier_split: Path, destination: Path) -> None:
    """Recopie un split en reindexant les classes locales vers COCO.

    Les images sont liees en dur (instantane, aucun espace disque duplique) ;
    seuls les fichiers de labels sont reecrits.
    """
    images_src = dossier_split / "images"
    labels_src = dossier_split / "labels"
    images_dst = destination / "images"
    labels_dst = destination / "labels"

    if destination.exists():
        shutil.rmtree(destination)
    images_dst.mkdir(parents=True)
    labels_dst.mkdir(parents=True)

    for image in sorted(images_src.iterdir()):
        if not image.is_file():
            continue
        cible = images_dst / image.name
        try:
            os.link(image, cible)
        except OSError:
            shutil.copy2(image, cible)

        label = labels_src / f"{image.stem}.txt"
        if not label.exists():
            continue
        lignes = []
        for ligne in label.read_text(encoding="utf-8").splitlines():
            champs = ligne.split()
            if len(champs) < 5:
                continue
            champs[0] = str(LOCAL_VERS_COCO[int(champs[0])])
            lignes.append(" ".join(champs))
        (labels_dst / label.name).write_text("\n".join(lignes) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Metriques globales via Ultralytics
# --------------------------------------------------------------------------

def metriques_globales(modele: YOLO, data_yaml: Path, split: str, device: str) -> dict:
    """mAP, precision et rappel calcules par le validateur d'Ultralytics."""
    resultats = modele.val(
        data=str(data_yaml),
        split=split,
        device=device,
        conf=0.001,      # convention de detection : seuil bas pour la courbe P/R
        iou=0.6,
        classes=COCO_VEHICULES if modele_est_coco(modele) else None,
        verbose=False,
        plots=False,
    )
    boite = resultats.box
    precision = float(boite.mp)
    rappel = float(boite.mr)
    f1 = 2 * precision * rappel / (precision + rappel) if (precision + rappel) else 0.0
    return {
        "mAP@0.5": float(boite.map50),
        "mAP@0.5:0.95": float(boite.map),
        "precision": precision,
        "recall": rappel,
        "f1": f1,
    }


# --------------------------------------------------------------------------
# Rappel par taille d'objet, calcul maison
# --------------------------------------------------------------------------

def iou_matrice(boites_a: np.ndarray, boites_b: np.ndarray) -> np.ndarray:
    """IoU entre deux jeux de boites au format xyxy absolu."""
    if len(boites_a) == 0 or len(boites_b) == 0:
        return np.zeros((len(boites_a), len(boites_b)))

    x1 = np.maximum(boites_a[:, None, 0], boites_b[None, :, 0])
    y1 = np.maximum(boites_a[:, None, 1], boites_b[None, :, 1])
    x2 = np.minimum(boites_a[:, None, 2], boites_b[None, :, 2])
    y2 = np.minimum(boites_a[:, None, 3], boites_b[None, :, 3])

    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aire_a = (boites_a[:, 2] - boites_a[:, 0]) * (boites_a[:, 3] - boites_a[:, 1])
    aire_b = (boites_b[:, 2] - boites_b[:, 0]) * (boites_b[:, 3] - boites_b[:, 1])
    union = aire_a[:, None] + aire_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def charger_verite_terrain(label: Path, largeur: int, hauteur: int) -> np.ndarray:
    """Lit un fichier YOLO normalise et renvoie les boites en xyxy absolu."""
    if not label.exists():
        return np.zeros((0, 4))
    boites = []
    for ligne in label.read_text(encoding="utf-8").splitlines():
        champs = ligne.split()
        if len(champs) < 5:
            continue
        cx, cy, w, h = (float(v) for v in champs[1:5])
        cx, w = cx * largeur, w * largeur
        cy, h = cy * hauteur, h * hauteur
        boites.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return np.array(boites) if boites else np.zeros((0, 4))


def rappel_par_taille(modele: YOLO, dossier_split: Path, device: str) -> tuple[dict, float]:
    """Rappel agnostique a la classe, ventile par aire de la boite reelle.

    Renvoie aussi le debit en images par seconde, mesure sur les memes images.
    """
    images = [p for p in sorted((dossier_split / "images").iterdir()) if p.is_file()]
    if not images:
        sys.exit(f"Aucune image dans {dossier_split / 'images'}")

    classes = COCO_VEHICULES if modele_est_coco(modele) else None
    compte = {nom: {"total": 0, "detecte": 0} for nom, _, _ in BORNES_TAILLE}
    duree_totale = 0.0

    # Echauffement : le premier appel paie l'initialisation du modele et du
    # device ; il ne doit pas entrer dans la mesure du debit.
    modele.predict(str(images[0]), conf=SEUIL_CONFIANCE, classes=classes,
                   device=device, verbose=False)

    for chemin in images:
        debut = time.perf_counter()
        prediction = modele.predict(
            str(chemin),
            conf=SEUIL_CONFIANCE,
            classes=classes,
            device=device,
            verbose=False,
        )[0]
        duree_totale += time.perf_counter() - debut

        hauteur, largeur = prediction.orig_shape
        reelles = charger_verite_terrain(
            dossier_split / "labels" / f"{chemin.stem}.txt", largeur, hauteur
        )
        predites = prediction.boxes.xyxy.cpu().numpy() if len(prediction.boxes) else np.zeros((0, 4))

        # Facteur de reduction applique par le reseau a cette image.
        echelle = TAILLE_REFERENCE / max(largeur, hauteur)

        ious = iou_matrice(reelles, predites)
        # Une prediction ne peut valider qu'une seule boite reelle.
        deja_prises: set[int] = set()
        for i in range(len(reelles)):
            aire = (reelles[i, 2] - reelles[i, 0]) * (reelles[i, 3] - reelles[i, 1])
            aire *= echelle**2
            for nom, mini, maxi in BORNES_TAILLE:
                if mini <= aire < maxi:
                    bucket = nom
                    break
            compte[bucket]["total"] += 1

            if len(predites) == 0:
                continue
            ordre = np.argsort(-ious[i])
            for j in ordre:
                if ious[i, j] < SEUIL_IOU:
                    break
                if j not in deja_prises:
                    deja_prises.add(int(j))
                    compte[bucket]["detecte"] += 1
                    break

    resultat = {}
    for nom, valeurs in compte.items():
        total = valeurs["total"]
        resultat[nom] = {
            "objets": total,
            "detectes": valeurs["detecte"],
            "rappel": valeurs["detecte"] / total if total else 0.0,
        }
    fps = len(images) / duree_totale if duree_totale else 0.0
    return resultat, fps


# --------------------------------------------------------------------------

def evaluer(chemin_modele: Path, data_yaml: Path, split: str, device: str) -> dict:
    if not chemin_modele.exists():
        sys.exit(f"Modele introuvable : {chemin_modele}")

    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    racine_dataset = (data_yaml.parent / config["path"]).resolve()
    dossier_split = racine_dataset / split
    if not (dossier_split / "images").exists():
        sys.exit(
            f"{dossier_split / 'images'} introuvable.\n"
            "Lancez d'abord : python scripts/import_dataset.py"
        )

    modele = YOLO(str(chemin_modele))
    est_coco = modele_est_coco(modele)
    print(f"Modele : {chemin_modele.name} ({len(modele.names)} classes, "
          f"{'indexation COCO' if est_coco else 'indexation locale'})")

    if est_coco:
        # Le validateur compare des indices de classes : on lui presente une vue
        # du split reindexee en COCO, sinon la comparaison est fausse.
        vue = racine_dataset / f"_vue_coco_{split}"
        construire_vue_coco(dossier_split, vue)
        yaml_temporaire = vue.parent / f"_vue_coco_{split}.yaml"
        # Ultralytics exige les cles train et val dans tout data.yaml, meme
        # lorsqu'on ne valide qu'un seul split : on les fait toutes pointer sur
        # la vue reindexee, seul `split` est reellement lu.
        chemin_vue = f"{vue.name}/images"
        yaml_temporaire.write_text(
            yaml.safe_dump(
                {
                    "path": str(vue.parent),
                    "train": chemin_vue,
                    "val": chemin_vue,
                    "test": chemin_vue,
                    "names": {int(k): v for k, v in modele.names.items()},
                },
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        globales = metriques_globales(modele, yaml_temporaire, split, device)
        tailles, fps = rappel_par_taille(modele, dossier_split, device)
        shutil.rmtree(vue, ignore_errors=True)
        yaml_temporaire.unlink(missing_ok=True)
    else:
        globales = metriques_globales(modele, data_yaml, split, device)
        tailles, fps = rappel_par_taille(modele, dossier_split, device)

    return {
        "modele": str(chemin_modele.relative_to(RACINE)),
        "split": split,
        "globales": globales,
        "fps": fps,
        "rappel_par_taille": tailles,
    }


def afficher(rapport: dict) -> None:
    g = rapport["globales"]
    print(f"\n--- {rapport['modele']} (split {rapport['split']}) ---")
    print(f"  mAP@0.5        : {g['mAP@0.5']:.4f}")
    print(f"  mAP@0.5:0.95   : {g['mAP@0.5:0.95']:.4f}")
    print(f"  Precision      : {g['precision']:.4f}")
    print(f"  Recall         : {g['recall']:.4f}")
    print(f"  F1-score       : {g['f1']:.4f}")
    print(f"  Debit          : {rapport['fps']:.1f} FPS")
    print("\n  Rappel par taille d'objet (convention COCO) :")
    for nom, v in rapport["rappel_par_taille"].items():
        print(f"    {nom:<7} {v['detectes']:>5}/{v['objets']:<5} = {v['rappel']:.4f}")


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--model", type=Path, default=RACINE / "models" / "yolov8n.pt")
    parseur.add_argument("--data", type=Path, default=CONFIG_DATASET)
    parseur.add_argument("--split", default="test", choices=["train", "val", "test"])
    parseur.add_argument("--device", default="")
    parseur.add_argument("--out", type=Path, help="fichier JSON de sortie")
    args = parseur.parse_args()

    chemin = args.model if args.model.is_absolute() else RACINE / args.model
    rapport = evaluer(chemin, args.data, args.split, args.device)
    afficher(rapport)

    DOSSIER_RESULTATS.mkdir(exist_ok=True)
    sortie = args.out or DOSSIER_RESULTATS / f"eval_{chemin.stem}_{args.split}.json"
    sortie.write_text(json.dumps(rapport, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapport ecrit dans {sortie}")


if __name__ == "__main__":
    main()
