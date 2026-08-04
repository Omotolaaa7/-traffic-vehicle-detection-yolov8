"""Etape 2 de docs/PROCEDURE.md : import d'un jeu de donnees telecharge.

Prend un jeu au format YOLO (export Roboflow, Kaggle, Hugging Face) et le
convertit vers data/dataset/ avec nos 4 classes alignees sur COCO.

Deux operations essentielles :

1. **Remappage des classes.** Un jeu telecharge a ses propres classes et son
   propre ordre. La validite de la comparaison avec le modele pre-entraine
   repose entierement sur l'alignement de nos classes avec COCO (voir
   LOCAL_VERS_COCO dans evaluate.py). Sans remappage, la comparaison est fausse.
2. **Garantie d'un split de test isole.** Si le jeu source n'en fournit pas, le
   script en derive un depuis la validation, de facon deterministe.

Lance sans correspondance valide, le script affiche les classes reellement
presentes dans le jeu source puis s'arrete : c'est le moyen le plus simple de
remplir configs/import.yaml.

Usage :
    python scripts/import_dataset.py
    python scripts/import_dataset.py --source data/raw/mon_dataset
"""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

import yaml

RACINE = Path(__file__).resolve().parent.parent
CONFIG_IMPORT = RACINE / "configs" / "import.yaml"
CONFIG_DATASET = RACINE / "configs" / "dataset.yaml"
DATASET = RACINE / "data" / "dataset"

CLASSES = ["voiture", "moto", "bus", "camion"]
INDICE = {nom: i for i, nom in enumerate(CLASSES)}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Noms de dossiers rencontres selon les exportateurs.
ALIAS_SPLITS = {
    "train": ["train", "training"],
    "val": ["val", "valid", "validation"],
    "test": ["test", "testing"],
}


def lire_classes_source(source: Path) -> list[str]:
    """Recupere la liste ordonnee des classes depuis le data.yaml du jeu."""
    candidats = list(source.glob("*.yaml")) + list(source.glob("*.yml"))
    if not candidats:
        sys.exit(
            f"Aucun fichier .yaml dans {source}.\n"
            "Un export YOLO contient un data.yaml decrivant les classes."
        )
    for chemin in candidats:
        contenu = yaml.safe_load(chemin.read_text(encoding="utf-8"))
        if isinstance(contenu, dict) and "names" in contenu:
            noms = contenu["names"]
            if isinstance(noms, dict):
                return [noms[k] for k in sorted(noms)]
            return list(noms)
    sys.exit(f"Aucune cle 'names' trouvee dans les YAML de {source}.")


def trouver_split(source: Path, split: str) -> Path | None:
    for alias in ALIAS_SPLITS[split]:
        for dossier in (source / alias, source / alias / "images"):
            if dossier.is_dir() and dossier.name == "images":
                return dossier.parent
            if dossier.is_dir() and (dossier / "images").is_dir():
                return dossier
    return None


def construire_table(classes_source: list[str], correspondance: dict) -> dict[int, int]:
    """Associe chaque indice source a un indice local, ou l'exclut."""
    table: dict[int, int] = {}
    inconnues, ignorees = [], []

    for i, nom in enumerate(classes_source):
        cible = correspondance.get(nom)
        if cible is None:
            inconnues.append(nom)
        elif cible == "ignorer":
            ignorees.append(nom)
        elif cible in INDICE:
            table[i] = INDICE[cible]
        else:
            sys.exit(
                f"Classe cible inconnue pour '{nom}' : '{cible}'.\n"
                f"Valeurs acceptees : {', '.join(CLASSES)}, ignorer."
            )

    print(f"\nClasses du jeu source ({len(classes_source)}) :")
    for i, nom in enumerate(classes_source):
        if i in table:
            print(f"  {nom:<20} -> {CLASSES[table[i]]}")
        elif nom in ignorees:
            print(f"  {nom:<20} -> ignoree (choix explicite)")
        else:
            print(f"  {nom:<20} -> NON MAPPEE")

    if inconnues:
        print(
            f"\n{len(inconnues)} classe(s) absente(s) de configs/import.yaml : "
            f"{', '.join(inconnues)}"
        )
        print("Ajoutez-les sous `correspondance`, en cible ou en `ignorer`.")
        sys.exit("Import interrompu : la correspondance est incomplete.")

    if not table:
        sys.exit("Aucune classe source ne correspond a nos 4 classes.")
    return table


def convertir_label(chemin: Path, table: dict[int, int]) -> list[str]:
    lignes = []
    for ligne in chemin.read_text(encoding="utf-8").splitlines():
        champs = ligne.split()
        if len(champs) < 5:
            continue
        source = int(float(champs[0]))
        if source not in table:
            continue
        champs[0] = str(table[source])
        lignes.append(" ".join(champs))
    return lignes


def copier_split(dossier_source: Path, destination: Path, table: dict[int, int]) -> dict:
    images_dst = destination / "images"
    labels_dst = destination / "labels"
    for dossier in (images_dst, labels_dst):
        if dossier.exists():
            shutil.rmtree(dossier)
        dossier.mkdir(parents=True)

    compte = {i: 0 for i in range(len(CLASSES))}
    nb_images = 0

    for image in sorted((dossier_source / "images").iterdir()):
        if image.suffix.lower() not in EXTENSIONS:
            continue
        label_src = dossier_source / "labels" / f"{image.stem}.txt"
        lignes = convertir_label(label_src, table) if label_src.exists() else []

        cible = images_dst / image.name
        try:
            os.link(image, cible)
        except OSError:
            shutil.copy2(image, cible)
        (labels_dst / f"{image.stem}.txt").write_text(
            "\n".join(lignes) + ("\n" if lignes else ""), encoding="utf-8"
        )

        nb_images += 1
        for ligne in lignes:
            compte[int(ligne.split()[0])] += 1

    return {"images": nb_images, "classes": compte}


def deriver_test_depuis_val(graine: int) -> None:
    """Coupe le split de validation en deux quand la source n'a pas de test.

    Sans jeu de test isole, il n'y a pas de resultat mesurable, seulement une
    demonstration. Mieux vaut une validation plus petite qu'aucun test.
    """
    val = DATASET / "val"
    test = DATASET / "test"
    for dossier in (test / "images", test / "labels"):
        if dossier.exists():
            shutil.rmtree(dossier)
        dossier.mkdir(parents=True)

    images = sorted((val / "images").iterdir())
    random.Random(graine).shuffle(images)
    for image in images[: len(images) // 2]:
        shutil.move(str(image), test / "images" / image.name)
        label = val / "labels" / f"{image.stem}.txt"
        if label.exists():
            shutil.move(str(label), test / "labels" / label.name)

    print(
        f"\nLe jeu source ne fournit pas de split de test : {len(images) // 2} images "
        f"ont ete prelevees sur la validation (graine {graine})."
    )


def ecrire_config_dataset() -> None:
    lignes = [
        "# Configuration du jeu de donnees au format Ultralytics.",
        "#",
        "# Genere par scripts/import_dataset.py.",
        "# Ne pas editer a la main : les modifications seront perdues.",
        "#",
        "# Les 4 classes sont alignees sur des classes COCO afin que le modele",
        "# pre-entraine et le modele fine-tune soient comparables sur le meme",
        "# jeu de test (voir LOCAL_VERS_COCO dans scripts/evaluate.py) :",
        "#   voiture -> COCO 2  (car)",
        "#   moto    -> COCO 3  (motorcycle)",
        "#   bus     -> COCO 5  (bus)",
        "#   camion  -> COCO 7  (truck)",
        "",
        "path: ../data/dataset",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        "",
        "names:",
    ]
    lignes += [f"  {i}: {nom}" for i, nom in enumerate(CLASSES)]
    CONFIG_DATASET.write_text("\n".join(lignes) + "\n", encoding="utf-8")


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--config", type=Path, default=CONFIG_IMPORT)
    parseur.add_argument("--source", type=Path, help="surcharge le chemin du fichier de config")
    parseur.add_argument("--seed", type=int, default=42)
    args = parseur.parse_args()

    if not args.config.exists():
        sys.exit(f"Configuration introuvable : {args.config}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    source = args.source or Path(config["source"])
    source = source if source.is_absolute() else RACINE / source
    if not source.is_dir():
        sys.exit(
            f"Jeu source introuvable : {source}\n"
            "Telechargez et decompressez le jeu, puis renseignez `source` "
            "dans configs/import.yaml."
        )

    classes_source = lire_classes_source(source)
    table = construire_table(classes_source, config.get("correspondance", {}))

    resume = {}
    for split in ("train", "val", "test"):
        dossier = trouver_split(source, split)
        if dossier is None:
            resume[split] = None
            continue
        resume[split] = copier_split(dossier, DATASET / split, table)

    if resume["train"] is None:
        sys.exit("Le jeu source ne contient aucun split d'entrainement exploitable.")
    if resume["val"] is None:
        sys.exit("Le jeu source ne contient aucun split de validation exploitable.")
    if resume["test"] is None:
        deriver_test_depuis_val(args.seed)

    ecrire_config_dataset()

    print("\nInstances importees par classe :\n")
    entete = f"{'Split':<8}{'Images':>8}" + "".join(f"{c:>10}" for c in CLASSES)
    print(entete)
    print("-" * len(entete))
    for split in ("train", "val", "test"):
        dossier = DATASET / split / "labels"
        if not dossier.exists():
            continue
        compte = {i: 0 for i in range(len(CLASSES))}
        nb = 0
        for label in dossier.iterdir():
            nb += 1
            for ligne in label.read_text(encoding="utf-8").splitlines():
                if ligne.strip():
                    compte[int(ligne.split()[0])] += 1
        ligne_txt = f"{split:<8}{nb:>8}" + "".join(f"{compte[i]:>10}" for i in range(len(CLASSES)))
        print(ligne_txt)

    print(f"\nEcrit dans {DATASET}")
    print(f"Configuration mise a jour : {CONFIG_DATASET}")
    print("\nVerifiez le tableau ci-dessus avant d'entrainer : une classe a moins")
    print("de quelques dizaines d'instances en test ne donnera pas de metrique fiable.")


if __name__ == "__main__":
    main()
