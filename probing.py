import torch
import pickle
import utils
import math
import random

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
    N = len(input_paths)

    # Raw data and label
    raw_data = []
    y = []
    for path in input_paths:
        with open(path, "rb") as f:
            data = pickle.load(f)
            for _ in range(M):
                idx = random.randint(0, len(data) - max_seq)
                control = sum(data[idx : idx + max_seq])
                raw_data.append(data[idx : idx + max_seq])
                y.append(label[path.split("/")[-1]] + [control])
    x = torch.Tensor(raw_data)
    y = torch.Tensor(y)

    # Set up first steps for models
    _, _, look_ahead_mask = utils.get_masked_with_pad_tensor(max_seq, x, x, pad_token)
    x = mt.Decoder.embedding(x.to(torch.long))
    x *= math.sqrt(emb_dim)
    x = mt.Decoder.pos_encoding(x)

    # Output per layer
    X = torch.zeros((num_layer, N * M, emb_dim))
    for idx in range(num_layer):
        x, _ = mt.Decoder.enc_layers[idx](x, look_ahead_mask)
        X[idx] = x.mean(dim=1).detach()

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
    X, y = probe_task(
        [
            "dataset/processed/MIDI-Unprocessed_01_R1_2006_01-09_ORIG_MID--AUDIO_01_R1_2006_01_Track01_wav.midi.pickle"
        ],
        "trained_models/8.pth",
        {
            "MIDI-Unprocessed_01_R1_2006_01-09_ORIG_MID--AUDIO_01_R1_2006_01_Track01_wav.midi.pickle": [
                1
            ]
        },
        10,
    )
    print(X.shape)
    # models = probe_classifier(X, y.to(torch.long), 3, 0.1, 10)
