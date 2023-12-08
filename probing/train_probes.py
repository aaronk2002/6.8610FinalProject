import torch
import pandas as pd
import argparse
from probing import probe_classifier, probe_regressor

# Dataframe
df = pd.read_csv("../dataset/maestro_new.csv")

# Label to idx
key_to_idx = {"major": 1, "minor": 0}
composers = df["canonical_composer"].unique()
composers = [composer for composer in composers if "/" not in composer]
composers.sort()
composer_to_idx = {composer: idx for idx, composer in enumerate(composers)}

# Get dataset
results = {
    idx: torch.load(
        f"../dataset/{idx}-layers-probe.pth", map_location=torch.device("cpu")
    )
    for idx in [4, 6, 8]
}

# Train models
key_models = {}
composer_models = {}
control_models = {}
for idx in [4, 6, 8]:
    print("Key Model")
    key_models[idx] = probe_classifier(
        results[idx]["train_x"],
        results[idx]["train_y"][:, 1].to(torch.long),
        len(key_to_idx),
        0.1,
        100000,
    )
    print("Composer Model")
    composer_models[idx] = probe_classifier(
        results[idx]["train_x"],
        results[idx]["train_y"][:, 0].to(torch.long),
        len(composer_to_idx),
        0.1,
        100000,
    )
    print("Control Model")
    control_models[idx] = probe_regressor(
        results[idx]["train_x"],
        results[idx]["train_y"][:, 0],
        0.1,
        100000,
    )

# Save models
torch.save(control_models, "control.pth")
torch.save(composer_models, "composer.pth")
torch.save(key_models, "key.pth")
