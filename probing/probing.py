import torch
import pickle
import sys

# setting path
sys.path.append("..")

import utils
import math
import random
import pandas as pd
from data import Data
import argparse
from tqdm import tqdm

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

    # Random number for control
    rand = torch.rand(pad_token + 1) * 2

    # Raw data and label
    raw_data = []
    y = []
    for path in input_paths:
        with open(path, "rb") as f:
            data = pickle.load(f)
            if len(data) < max_seq:
                continue
            for idx in range(min(M, len(data) - max_seq + 1)):
                control = (
                    sum([rand[num].item() for num in data[idx : idx + max_seq]])
                    / max_seq
                )
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
    models = [torch.nn.Linear(d, num_classes) for _ in range(num_model)]
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train num_model models
    for idx in range(num_model):
        model = models[idx]
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        for _ in tqdm(range(epochs)):
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

    # Train num_model models
    for idx in range(num_model):
        model = models[idx]
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        for _ in tqdm(range(epochs)):
            pred = model(X[idx])
            optimizer.zero_grad()
            loss = loss_fn(pred.reshape(-1), y)
            if idx == 2:
                print(loss)
            loss.backward()
            optimizer.step()
    return models


if __name__ == "__main__":
    result = torch.load(
        f"../dataset/4-layers-probe.pth", map_location=torch.device("cpu")
    )
    probe_regressor(result["train_x"], result["train_y"][:, 2], 0.1, 10000)
