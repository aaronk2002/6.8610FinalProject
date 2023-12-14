import torch
from data import Data
import utils
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--layers", help="number of layers", type=int)
parser.add_argument("--N", help="number of samples in a batch", type=int)
parser.add_argument("--M", help="number of batches", type=int)
args = parser.parse_args()

# config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pad_token = 388
data_files = "dataset/processed"
model_path = f"trained_models/{args.layers}.pth"
N = args.N
M = args.M

# Data and Model
dataset = Data(data_files)
mt = torch.load(model_path, map_location=torch.device("cpu")).to(device)
mt.eval()

# Set up means
train_nll_mean = 0
eval_nll_mean = 0
train_acc_mean = 0
eval_acc_mean = 0

with torch.no_grad():
    for _ in tqdm(range(M)):
        # Batch train
        train_x, train_y = dataset.slide_seq2seq_batch(N, mt.max_seq)
        train_x = (
            torch.from_numpy(train_x)
            .contiguous()
            .to(device, non_blocking=True, dtype=torch.int)
        )
        train_y = (
            torch.from_numpy(train_y)
            .contiguous()
            .to(device, non_blocking=True, dtype=torch.long)
        )

        # Batch eval
        eval_x, eval_y = dataset.slide_seq2seq_batch(N, mt.max_seq, "eval")
        eval_x = (
            torch.from_numpy(eval_x)
            .contiguous()
            .to(device, non_blocking=True, dtype=torch.int)
        )
        eval_y = (
            torch.from_numpy(eval_y)
            .contiguous()
            .to(device, non_blocking=True, dtype=torch.long)
        )

        # Forward train
        _, _, look_ahead_mask = utils.get_masked_with_pad_tensor(
            mt.max_seq, train_x, train_x, pad_token
        )
        decoder, _ = mt.Decoder(train_x, mask=look_ahead_mask)
        y_pred_train = mt.fc(decoder)[:, -1, :]
        y_train = train_y[:, -1]

        # Forward eval
        _, _, look_ahead_mask = utils.get_masked_with_pad_tensor(
            mt.max_seq, eval_x, eval_x, pad_token
        )
        decoder, _ = mt.Decoder(eval_x, mask=look_ahead_mask)
        y_pred_eval = mt.fc(decoder)[:, -1, :]
        y_eval = eval_y[:, -1]

        # Evaluate
        loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
        train_nll = loss_fn(y_pred_train, y_train) / N
        eval_nll = loss_fn(y_pred_eval, y_eval) / N
        train_acc = torch.sum((torch.argmax(y_pred_train, dim=1) == y_train) / N).item()
        eval_acc = torch.sum((torch.argmax(y_pred_eval, dim=1) == y_eval) / N).item()

        # Update Mean
        train_nll_mean += train_nll / M
        eval_nll_mean += eval_nll / M
        train_acc_mean += train_acc / M
        eval_acc_mean += eval_acc / M

train_perplexity = torch.exp(train_nll_mean)
eval_perplexity = torch.exp(eval_nll_mean)
print(
    f"train nll = {train_nll_mean.item()}, train perplexity = {train_perplexity.item()}, train accuracy = {train_acc_mean}"
)
print(
    f"eval nll = {eval_nll_mean.item()}, eval perplexity = {eval_perplexity.item()}, eval accuracy = {eval_acc_mean}"
)
