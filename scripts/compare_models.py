"""Etape 6 de docs/PROCEDURE.md : YOLOv8 pre-entraine contre YOLOv8 fine-tune.

Evalue les deux modeles sur le meme split de test et produit le tableau
comparatif du rapport, au format Markdown et CSV.

C'est ce tableau qui constitue le resultat du projet : il chiffre l'ecart
entre un modele entraine ailleurs et le meme modele adapte au trafic beninois.

Usage :
    python scripts/compare_models.py
    python scripts/compare_models.py --finetuned models/finetuned/yolov8n_benin.pt
"""

import argparse
import csv
import json
from pathlib import Path

from evaluate import CONFIG_DATASET, DOSSIER_RESULTATS, RACINE, evaluer

LIGNES_GLOBALES = [
    ("mAP@0.5", "mAP@0.5", "{:.4f}"),
    ("mAP@0.5:0.95", "mAP@0.5:0.95", "{:.4f}"),
    ("Precision", "precision", "{:.4f}"),
    ("Recall", "recall", "{:.4f}"),
    ("F1-score", "f1", "{:.4f}"),
]


def ecart_relatif(avant: float, apres: float) -> str:
    if avant == 0:
        return "n/a" if apres == 0 else "+inf"
    return f"{(apres - avant) / avant * 100:+.1f} %"


def construire_lignes(base: dict, finetune: dict) -> list[tuple[str, str, str, str]]:
    lignes = []
    for libelle, cle, fmt in LIGNES_GLOBALES:
        a = base["globales"][cle]
        b = finetune["globales"][cle]
        lignes.append((libelle, fmt.format(a), fmt.format(b), ecart_relatif(a, b)))

    lignes.append(
        ("Debit (FPS)", f"{base['fps']:.1f}", f"{finetune['fps']:.1f}",
         ecart_relatif(base["fps"], finetune["fps"]))
    )

    for taille in ("petit", "moyen", "grand"):
        a = base["rappel_par_taille"][taille]
        b = finetune["rappel_par_taille"][taille]
        lignes.append((
            f"Rappel objets {taille}s ({a['objets']} objets)",
            f"{a['rappel']:.4f}",
            f"{b['rappel']:.4f}",
            ecart_relatif(a["rappel"], b["rappel"]),
        ))
    return lignes


def ecrire_markdown(lignes, base, finetune, split, chemin: Path) -> None:
    texte = [
        "# Comparaison YOLOv8 pre-entraine contre YOLOv8 fine-tune",
        "",
        f"Split evalue : `{split}`",
        f"Modele de reference : `{base['modele']}`",
        f"Modele fine-tune : `{finetune['modele']}`",
        "",
        "| Metrique | Pre-entraine | Fine-tune | Ecart |",
        "|---|---|---|---|",
    ]
    texte += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in lignes]
    texte += [
        "",
        "Le rappel par taille d'objet suit la convention COCO : petit sous 32x32",
        "pixels, moyen sous 96x96, grand au-dela. C'est la ligne des objets petits",
        "qui repond directement a la problematique des vehicules eloignes.",
        "",
    ]
    chemin.write_text("\n".join(texte), encoding="utf-8")


def ecrire_csv(lignes, chemin: Path) -> None:
    with chemin.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metrique", "pre_entraine", "fine_tune", "ecart"])
        writer.writerows(lignes)


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--baseline", type=Path, default=Path("models/yolov8n.pt"))
    parseur.add_argument(
        "--finetuned", type=Path, default=Path("models/finetuned/yolov8n_benin.pt")
    )
    parseur.add_argument("--data", type=Path, default=CONFIG_DATASET)
    parseur.add_argument("--split", default="test", choices=["train", "val", "test"])
    parseur.add_argument("--device", default="")
    args = parseur.parse_args()

    base_path = args.baseline if args.baseline.is_absolute() else RACINE / args.baseline
    fine_path = args.finetuned if args.finetuned.is_absolute() else RACINE / args.finetuned

    print("=== Modele pre-entraine ===")
    base = evaluer(base_path, args.data, args.split, args.device)
    print("\n=== Modele fine-tune ===")
    finetune = evaluer(fine_path, args.data, args.split, args.device)

    lignes = construire_lignes(base, finetune)

    largeur = max(len(l[0]) for l in lignes)
    print(f"\n{'Metrique'.ljust(largeur)} {'Pre-entr.':>12} {'Fine-tune':>12} {'Ecart':>10}")
    print("-" * (largeur + 38))
    for libelle, a, b, ecart in lignes:
        print(f"{libelle.ljust(largeur)} {a:>12} {b:>12} {ecart:>10}")

    DOSSIER_RESULTATS.mkdir(exist_ok=True)
    ecrire_markdown(lignes, base, finetune, args.split, DOSSIER_RESULTATS / "comparaison.md")
    ecrire_csv(lignes, DOSSIER_RESULTATS / "comparaison.csv")
    (DOSSIER_RESULTATS / "comparaison.json").write_text(
        json.dumps({"pre_entraine": base, "fine_tune": finetune}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nEcrit dans {DOSSIER_RESULTATS}/ : comparaison.md, comparaison.csv, comparaison.json")


if __name__ == "__main__":
    main()
