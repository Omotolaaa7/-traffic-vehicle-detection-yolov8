"""Telecharge un sous-ensemble BMD-45 depuis Hugging Face sans tout recuperer.

Le script lit le dataset en streaming (pas de chargement complet en RAM), puis
n'enregistre localement qu'un nombre cible d'images/labels au format YOLO.

Sortie par defaut:
  data/raw/bmd45_subset/
    train/images, train/labels
    val/images,   val/labels
    data.yaml

Usage:
  python scripts/download_bmd45_subset.py
  python scripts/download_bmd45_subset.py --train 2400 --val 600 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset, load_dataset_builder

RACINE = Path(__file__).resolve().parent.parent
DESTINATION_DEFAUT = RACINE / "data" / "raw" / "bmd45_subset"
REPO = "iisc-aim/BMD-45"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default=REPO, help="Dataset Hugging Face (defaut: iisc-aim/BMD-45)")
    p.add_argument("--dest", type=Path, default=DESTINATION_DEFAUT, help="Dossier de sortie")
    p.add_argument("--train", type=int, default=2400, help="Nombre d'images train a garder")
    p.add_argument("--val", type=int, default=600, help="Nombre d'images val a garder")
    p.add_argument("--seed", type=int, default=42, help="Graine du melange")
    p.add_argument(
        "--shuffle-buffer",
        type=int,
        default=10000,
        help="Taille du buffer de melange (IterableDataset)",
    )
    p.add_argument(
        "--image-format",
        default="jpg",
        choices=["jpg", "png"],
        help="Format d'image a ecrire localement",
    )
    return p.parse_args()


def get_class_names(repo: str) -> list[str]:
    """Recupere les noms de classes depuis les metadata HF si disponibles."""
    builder = load_dataset_builder(repo)
    objects_feature = builder.info.features["objects"]

    # Les datasets HF varient selon les clefs exposees: category, categories,
    # label(s). On essaie les variantes les plus courantes.
    for key in ("category", "categories", "label", "labels"):
        if key in objects_feature:
            feature = objects_feature[key]
            names = list(getattr(feature, "names", []) or [])
            if names:
                return names

    if repo == "iisc-aim/BMD-45":
        return [
            "Hatchback",
            "Sedan",
            "SUV",
            "MUV",
            "Bus",
            "Truck",
            "Three-wheeler",
            "Two-wheeler",
            "LCV",
            "Mini-bus",
            "Tempo-traveller",
            "Bicycle",
            "Van",
            "Other",
        ]

    raise RuntimeError(
        "Impossible de lire les noms de classes depuis le dataset. "
        f"Cles disponibles dans objects: {list(objects_feature.keys())}"
    )


def extract_categories(objects: dict) -> list[int]:
    """Recupere les IDs de classes quelle que soit la cle du schema HF."""
    for key in ("category", "categories", "label", "labels"):
        values = objects.get(key)
        if values is not None:
            return [int(v) for v in values]
    return []


def yolo_line(category_id: int, bbox_xywh: list[float], width: int, height: int) -> str:
    x, y, w, h = bbox_xywh
    cx = (x + w / 2.0) / width
    cy = (y + h / 2.0) / height
    nw = w / width
    nh = h / height

    # Clamp defensif pour rester valide si une boite depasse de 1-2 px.
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    nw = min(max(nw, 0.0), 1.0)
    nh = min(max(nh, 0.0), 1.0)

    return f"{category_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def compter_paires_completes(
    images_dir: Path, labels_dir: Path, split_name: str, image_format: str
) -> int:
    """Compte les paires image+label deja exportees lors d'une session precedente.

    Les fichiers sont nommes sequentiellement : on avance tant que la paire est
    complete. Une paire incomplete (interruption entre l'ecriture de l'image et
    celle du label) est supprimee pour repartir d'un etat propre.
    """
    n = 0
    while True:
        stem = f"{split_name}_{n:06d}"
        image = images_dir / f"{stem}.{image_format}"
        label = labels_dir / f"{stem}.txt"
        if image.exists() and label.exists():
            n += 1
            continue
        image.unlink(missing_ok=True)
        label.unlink(missing_ok=True)
        return n


def export_split(
    repo: str,
    hf_split: str,
    split_name: str,
    n_images: int,
    seed: int,
    shuffle_buffer: int,
    destination: Path,
    image_format: str,
) -> int:
    images_dir = destination / split_name / "images"
    labels_dir = destination / split_name / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Reprise apres interruption : les paires deja presentes sont conservees.
    # Le flux etant melange avec la meme graine, l'ordre des echantillons est
    # identique d'une session a l'autre : on saute ceux deja exportes.
    deja = compter_paires_completes(images_dir, labels_dir, split_name, image_format)
    if deja >= n_images:
        print(f"{split_name}: deja complet ({deja}/{n_images}), rien a telecharger")
        return deja
    if deja:
        print(f"{split_name}: reprise, {deja}/{n_images} images deja presentes")

    ds = load_dataset(repo, split=hf_split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    count = deja
    for i, sample in enumerate(ds):
        if count >= n_images:
            break
        if i < deja:
            continue

        image = sample["image"]
        objects = sample["objects"]

        width, height = image.size
        stem = f"{split_name}_{i:06d}"
        image_path = images_dir / f"{stem}.{image_format}"
        label_path = labels_dir / f"{stem}.txt"

        if image_format == "jpg":
            # PIL re-encode a l'enregistrement ; la qualite par defaut (~75)
            # degraderait visiblement les petits objets, coeur du sujet.
            image.save(image_path, quality=95)
        else:
            image.save(image_path)

        bboxes = objects.get("bbox", [])
        categories = extract_categories(objects)
        lines = []
        for cat_id, bbox in zip(categories, bboxes):
            lines.append(yolo_line(cat_id, bbox, width, height))

        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        count += 1

        if count % 200 == 0:
            print(f"{split_name}: {count}/{n_images}")

    return count


def write_data_yaml(destination: Path, names: list[str], has_test: bool) -> None:
    lines = [
        "# Genere par scripts/download_bmd45_subset.py",
        "path: .",
        "train: train/images",
        "val: val/images",
    ]
    if has_test:
        lines.append("test: test/images")

    lines.extend(["", "names:"])
    for i, name in enumerate(names):
        lines.append(f"  {i}: {name}")

    (destination / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    destination = args.dest if args.dest.is_absolute() else (RACINE / args.dest)
    destination.mkdir(parents=True, exist_ok=True)

    print("Lecture des classes...")
    names = get_class_names(args.repo)
    print(f"{len(names)} classes detectees: {', '.join(names)}")

    print("\nExport train...")
    n_train = export_split(
        repo=args.repo,
        hf_split="train",
        split_name="train",
        n_images=args.train,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
        destination=destination,
        image_format=args.image_format,
    )

    print("\nExport val...")
    n_val = export_split(
        repo=args.repo,
        hf_split="val",
        split_name="val",
        n_images=args.val,
        seed=args.seed + 1,
        shuffle_buffer=args.shuffle_buffer,
        destination=destination,
        image_format=args.image_format,
    )

    write_data_yaml(destination, names, has_test=False)

    print("\nTermine.")
    print(f"Dossier: {destination}")
    print(f"Train: {n_train} images")
    print(f"Val  : {n_val} images")
    print("Test : non exporte (import_dataset.py pourra le deriver depuis val)")


if __name__ == "__main__":
    main()
