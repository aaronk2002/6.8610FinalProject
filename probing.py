import torch
import pickle
import utils
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pad_token = 388


def probe_task(input_paths, model_path, label):
    """
    Given the inputs, the trained transformer model, and the label, generate
    several datasets for the probing tasks, one dataset for each layer
    of the model

    Inputs:
    - input_paths   : list of paths to all midi files that will be used,
                      each as a forward slash separated string
    - model_path    : file adress for the transformer model
    - label         : dictionary mapping midi filename to the desired
                      label, and it must contain all midi filenames in
                      input_paths

    Return:
    torch tensor with size num_layer x N x d for input and torch tensor
    with size N for label, N is number of samples, d is length of samples,
    and num_layer is the number of layers
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
            raw_data.append(data[:max_seq])
            y.append(label[path.split("/")[-1]])
    x = torch.Tensor(raw_data)
    y = torch.Tensor(y)

    # Set up first steps for models
    _, _, look_ahead_mask = utils.get_masked_with_pad_tensor(max_seq, x, x, pad_token)
    x = mt.Decoder.embedding(x.to(torch.long))
    x *= math.sqrt(emb_dim)
    x = mt.Decoder.pos_encoding(x)

    # Output per layer
    X = torch.zeros((num_layer, N, emb_dim * max_seq))
    for idx in range(num_layer):
        x, _ = mt.Decoder.enc_layers[idx](x, look_ahead_mask)
        X[idx] = x.reshape(N, -1)

    return X, y


if __name__ == "__main__":
    X, y = probe_task(
        [
            "dataset/processed/MIDI-Unprocessed_01_R1_2006_01-09_ORIG_MID--AUDIO_01_R1_2006_01_Track01_wav.midi.pickle"
        ],
        "trained_models/8.pth",
        {
            "MIDI-Unprocessed_01_R1_2006_01-09_ORIG_MID--AUDIO_01_R1_2006_01_Track01_wav.midi.pickle": 1
        },
    )
    print(X.shape)
