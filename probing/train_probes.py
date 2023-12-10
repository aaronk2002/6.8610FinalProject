import torch
import pandas as pd
import argparse
from probing import probe_classifier, probe_regressor

### GLOABL VARIABLES ###
CONTROL_PATH = 'control.pth'
COMPOSER_PATH = 'composer.pth'
KEY_PATH = 'key.pth'
LR = 0.1
EPOCHS = 100000
# Dataframe
df = pd.read_csv("../dataset/maestro_new.csv")

### GET ARGUMENTS ###
parser = argparse.ArgumentParser()
parser.add_argument("--num_layers", help="number of decoder layers", type=str)
parser.add_argument("--control", help="Control type (key, composer, control)", type=str)
args = parser.parse_args()
control, num_layers = args.control, args.num_layers

print(f'num_layers = {num_layers} control = {control} ')

# Label to idx
key_to_idx = {"major": 1, "minor": 0}
composers = df["canonical_composer"].unique()
composers = [composer for composer in composers if "/" not in composer]
composers.sort()
composer_to_idx = {composer: idx for idx, composer in enumerate(composers)}

# Get dataset with `num_layers` layers
result = torch.load('f../dataset/{num_layers}-layers-probe.pth', map_location=torch.device("cpu"))

# Train models
if control == 'key':
    key_model = probe_classifier(
        result['train_x'],
        result['train_y'][:, 1].to(torch.long),
        len(key_to_idx),
        LR,
        EPOCHS,
    )
    torch.save(key_model, KEY_PATH)
elif control == 'composer':
    composer_model = probe_classifier(
        result['train_x'],
        result['train_y'][:, 0].to(torch.long),
        len(composer_to_idx),
        LR,
        EPOCHS,
    )
    torch.save(composer_model, COMPOSER_PATH)
elif control == 'control':
    control_model = probe_regressor(
        result['train_x'],
        result['train_y'][:, 2],
        LR,
        EPOCHS,
    )
    torch.save(control_model, CONTROL_PATH)