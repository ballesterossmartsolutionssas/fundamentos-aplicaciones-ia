from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

TARGET_COLUMN = "NObeyesdad"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = fetch_ucirepo(id=544)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    df = pd.concat([features, targets], axis=1)

    csv_path = output_dir / "obesity.csv"
    metadata_path = output_dir / "obesity_metadata.json"
    variables_path = output_dir / "obesity_variables.csv"

    df.to_csv(csv_path, index=False)

    metadata = {}
    if dataset.metadata is not None:
        metadata = dict(dataset.metadata)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if dataset.variables is not None:
        dataset.variables.to_csv(variables_path, index=False)

    print(f"Archivo guardado en: {csv_path}")
    print(f"Metadatos guardados en: {metadata_path}")
    print(f"Variables guardadas en: {variables_path}")
    print(f"Forma del dataset: {df.shape}")
    print(f"Clases objetivo: {sorted(df[TARGET_COLUMN].unique().tolist())}")


if __name__ == "__main__":
    main()
