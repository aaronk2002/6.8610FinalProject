import torch
import pandas as pd
import sys
import argparse

# setting path
sys.path.append("..")

from data import Data
from probing import probe_task

# Get arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model path", type=str)
parser.add_argument("--save", help="saving path", type=str)
args = parser.parse_args()
print(f"model = {args.model}, save = {args.save}")

# Dataframe
df = pd.read_csv("../dataset/maestro_new.csv")

# Label to idx
key_to_idx = {"major": 1, "minor": 0}
composers = df["canonical_composer"].unique()
composers = [composer for composer in composers if "/" not in composer]
composers.sort()
composer_to_idx = {composer: idx for idx, composer in enumerate(composers)}

# File to label
label = {}
for idx, row in df.iterrows():
    if "/" in row["canonical_composer"]:
        continue
    label[row["midi_filename"][5:] + ".pickle"] = [
        composer_to_idx[row["canonical_composer"]],
        key_to_idx[row["key_mode"]],
    ]

# Processing paths
dataset = Data("../dataset/processed")
train_path = [
    file.split("\\")[-1].split("/")[-1] for file in dataset.file_dict["train"]
]
train_path = ["../dataset/processed/" + file for file in train_path if file in label]
eval_path = [file.split("\\")[-1].split("/")[-1] for file in dataset.file_dict["eval"]]
eval_path = ["../dataset/processed/" + file for file in eval_path if file in label]

# Get probe datasets
train_x, train_y = probe_task(train_path, args.model, label, 1)
eval_x, eval_y = probe_task(eval_path, args.model, label, 1)
torch.save(
    {"train_x": train_x, "train_y": train_y, "eval_x": eval_x, "eval_y": eval_y},
    args.save,
)
