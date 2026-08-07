"""Trace les courbes d'entrainement de l'article a partir de results.csv.

Ultralytics produit deja un results.png, mais son texte est trop petit une
fois l'image reduite a la largeur d'une colonne, et le rendu matriciel
pixellise a l'impression. On retrace donc les memes donnees en PDF
vectoriel, avec les seules courbes commentees dans l'article.

Le CSV lu est celui du run de fine-tuning, dont l'emplacement decoule de
configs/entrainement.yaml (projet: results/entrainements, nom_run:
yolov8n_benin). Aucun chemin n'est code en dur ailleurs.

Usage :
    python scripts/tracer_courbes.py
    python scripts/tracer_courbes.py --run results/entrainements/yolov8n_benin
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pas d'affichage interactif : on ecrit un fichier
import matplotlib.pyplot as plt  # noqa: E402  (apres le choix du backend)

RACINE = Path(__file__).resolve().parent.parent
RUN_PAR_DEFAUT = RACINE / "results" / "entrainements" / "yolov8n_benin"
SORTIE_PAR_DEFAUT = RACINE / "article" / "figures" / "courbes_entrainement.pdf"

# Colonnes du results.csv d'Ultralytics, avec le libelle utilise dans la
# legende de la figure de l'article.
# Les libelles portent les accents : ils sont rendus dans la figure d'un
# article en francais, contrairement aux commentaires du code.
PERTES = [
    ("train/box_loss", "Boîte (entraînement)", "-"),
    ("train/cls_loss", "Classification (entraînement)", "-"),
    ("train/dfl_loss", "DFL (entraînement)", "-"),
    ("val/box_loss", "Boîte (validation)", "--"),
    ("val/cls_loss", "Classification (validation)", "--"),
    ("val/dfl_loss", "DFL (validation)", "--"),
]
METRIQUES = [
    ("metrics/mAP50(B)", "mAP@0,5"),
    ("metrics/mAP50-95(B)", "mAP@0,5:0,95"),
]


def lire_csv(chemin: Path) -> dict[str, list[float]]:
    """Retourne les colonnes du results.csv sous forme de listes de flottants.

    Ultralytics aligne les en-tetes avec des espaces : on les retire, sans
    quoi 'metrics/mAP50(B)' ne serait jamais trouve.
    """
    if not chemin.exists():
        raise SystemExit(
            f"Fichier introuvable : {chemin}\n"
            "Recuperez results.csv du run d'entrainement (voir docs/PROCEDURE.md)."
        )
    with chemin.open(newline="", encoding="utf-8") as flux:
        lignes = list(csv.DictReader(flux))
    if not lignes:
        raise SystemExit(f"{chemin} est vide.")
    colonnes: dict[str, list[float]] = {}
    for cle in lignes[0]:
        propre = cle.strip()
        colonnes[propre] = [float(ligne[cle]) for ligne in lignes]
    return colonnes


def tracer(colonnes: dict[str, list[float]], sortie: Path) -> None:
    epochs = colonnes["epoch"]
    figure, (gauche, droite) = plt.subplots(1, 2, figsize=(10, 3.8))

    for cle, libelle, style in PERTES:
        if cle in colonnes:
            gauche.plot(epochs, colonnes[cle], style, linewidth=1.4, label=libelle)
    gauche.set_xlabel("Epoch")
    gauche.set_ylabel("Perte")
    gauche.set_title("(a) Fonctions de perte")
    gauche.set_xlim(left=1)
    gauche.legend(fontsize=7, ncol=2)
    gauche.grid(alpha=0.3)

    for cle, libelle in METRIQUES:
        if cle in colonnes:
            droite.plot(epochs, colonnes[cle], linewidth=1.6, label=libelle)
    # Le maximum de mAP@0,5:0,95 est commente dans l'article : on le repere.
    if "metrics/mAP50-95(B)" in colonnes:
        valeurs = colonnes["metrics/mAP50-95(B)"]
        meilleure = max(range(len(valeurs)), key=valeurs.__getitem__)
        droite.axvline(epochs[meilleure], color="grey", linestyle=":", linewidth=1)
        droite.annotate(
            f"max epoch {int(epochs[meilleure])}\n{valeurs[meilleure]:.3f}".replace(".", ","),
            xy=(epochs[meilleure], valeurs[meilleure]),
            xytext=(-10, -34),
            textcoords="offset points",
            fontsize=7,
            ha="right",
        )
    droite.set_xlabel("Epoch")
    droite.set_ylabel("mAP (jeu de validation)")
    droite.set_title("(b) Précision moyenne")
    droite.set_xlim(left=1)
    droite.legend(fontsize=8, loc="lower right")
    droite.grid(alpha=0.3)

    figure.tight_layout()
    sortie.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(sortie, bbox_inches="tight")
    plt.close(figure)
    # Chemin relatif quand la sortie est dans le projet, absolu sinon
    # (--sortie accepte n'importe quel emplacement).
    try:
        affichage = sortie.relative_to(RACINE)
    except ValueError:
        affichage = sortie
    print(f"Figure ecrite : {affichage}")


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--run", type=Path, default=RUN_PAR_DEFAUT)
    parseur.add_argument("--sortie", type=Path, default=SORTIE_PAR_DEFAUT)
    args = parseur.parse_args()

    run = args.run if args.run.is_absolute() else RACINE / args.run
    sortie = args.sortie if args.sortie.is_absolute() else RACINE / args.sortie
    tracer(lire_csv(run / "results.csv"), sortie)


if __name__ == "__main__":
    main()
