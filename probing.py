import torch
import pickle
import utils
import math
import random
import pandas as pd
from data import Data
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pad_token = 388
random.seed(1)


def probe_task(input_paths, model_path, label, M):
    """
    Given the inputs, the trained transformer model, and the label, generate
    several datasets for the probing tasks, one dataset for each layer
    of the model

    Inputs:
    - input_paths   : list of paths to all midi files that will be used,
                      each as a forward slash separated string
    - model_path    : file adress for the transformer model
    - label         : dictionary mapping midi filename to the desired
                      label (list with d' elements), and it must contain
                      all midi filenames in input_paths
    - M             : number of samples per file

    Return:
    torch tensor with size num_layer x N*M x d for input and torch tensor
    with size N*M x (d'+1) for label, where d is length of samples, num_layer
    is the number of layers, N is the number of files in the input_paths,
    and d' is the dimensionality of each label, extra one for control
    """
    # Model information
    mt = torch.load(model_path, map_location=torch.device("cpu")).to(device)
    max_seq = mt.max_seq
    num_layer = mt.num_layer
    emb_dim = mt.embedding_dim

    # Raw data and label
    raw_data = []
    y = []
    for path in input_paths:
        with open(path, "rb") as f:
            data = pickle.load(f)
            if len(data) < max_seq:
                continue
            for idx in range(min(M, len(data) - max_seq + 1)):
                control = sum(data[idx : idx + max_seq])
                raw_data.append(data[idx : idx + max_seq])
                y.append(label[path.split("/")[-1]] + [control])
    x = torch.Tensor(raw_data).to(device)
    y = torch.Tensor(y).to(device)

    # Calculate outputs for every decoder layer
    X = torch.zeros((num_layer, len(x), emb_dim))
    for idx in range(len(x)):
        curr_x = x[idx : idx + 1]
        # Set up first steps for models
        _, _, look_ahead_mask = utils.get_masked_with_pad_tensor(
            max_seq, curr_x, curr_x, pad_token
        )
        curr_x = mt.Decoder.embedding(curr_x.to(torch.long))
        curr_x *= math.sqrt(emb_dim)
        curr_x = mt.Decoder.pos_encoding(curr_x)
        # Output per layer
        for jdx in range(num_layer):
            curr_x, _ = mt.Decoder.enc_layers[jdx](curr_x, look_ahead_mask)
            X[jdx, idx : idx + 1] = curr_x.mean(dim=1).detach()

    return X, y


def probe_classifier(X, y, num_classes, lr, epochs):
    """
    Given a set of input dataset and output, create a linear classifier
    model for each set of inputs

    Input:
    - X             : inputs with size num_model x N x d
    - y             : labels with size N
    - num_classes   : number of classes
    - lr            : learning step for optimization
    - epochs        : number of epochs

    Output:
    list of num_model models
    """
    # Initialization
    num_model, N, d = X.shape
    models = [
        torch.nn.Sequential(torch.nn.Linear(d, num_classes), torch.nn.LogSoftmax(dim=1))
        for _ in range(num_model)
    ]
    loss_fn = torch.nn.NLLLoss()

    # Train N models
    for idx in range(N):
        model = models[idx]
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        model.train()
        for _ in range(epochs):
            logits = model(X[idx])
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
    return models


def probe_regressor(X, y, lr, epochs):
    """
    Given a set of input dataset and output, create a linear regressor
    model for each set of inputs

    Input:
    - X             : inputs with size num_model x N x d
    - y             : labels with size N

    Output:
    list of num_model models
    """
    # Initialization
    num_model, N, d = X.shape
    models = [torch.nn.Linear(d, 1) for _ in range(num_model)]
    loss_fn = torch.nn.MSELoss()

    # Train N models
    for idx in range(N):
        model = models[idx]
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        model.train()
        for _ in range(epochs):
            pred = model(X[idx])
            optimizer.zero_grad()
            loss = loss_fn(pred.reshape(-1), y)
            loss.backward()
            optimizer.step()
    return models


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="model path", type=str)
    parser.add_argument("--save", help="saving path", type=str)
    args = parser.parse_args()
    print(f"model = {args.model}, save = {args.save}")

    # Dataframe
    df = pd.read_csv("dataset/maestro_new.csv")

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
    dataset = Data("dataset/processed")
    train_path = [
        file.split("\\")[-1].split("/")[-1] for file in dataset.file_dict["train"]
    ]
    train_path = ["dataset/processed/" + file for file in train_path if file in label]
    eval_path = [
        file.split("\\")[-1].split("/")[-1] for file in dataset.file_dict["eval"]
    ]
    eval_path = ["dataset/processed/" + file for file in eval_path if file in label]

    # Get probe datasets
    train_x, train_y = probe_task(train_path[:10], args.model, label, 1)
    eval_x, eval_y = probe_task(eval_path[:10], args.model, label, 1)
    torch.save(
        {"train_x": train_x, "train_y": train_y, "eval_x": eval_x, "eval_y": eval_y},
        args.save,
    )
