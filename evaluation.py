import torch
from data import Data
import utils

# config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pad_token = 388
data_files = "dataset/processed"
model_path = "trained_models/4.pth"
N = 100

# Data and Model
dataset = Data(data_files)
mt = torch.load(model_path, map_location=torch.device("cpu")).to(device)
mt.eval()

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
    torch.from_numpy(eval_x).contiguous().to(device, non_blocking=True, dtype=torch.int)
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
train_perplexity = torch.exp(train_nll)
eval_perplexity = torch.exp(eval_nll)
print(f"train nll = {train_nll.item()}, train perplexity = {train_perplexity.item()}")
print(f"eval nll = {eval_nll.item()}, eval perplexity = {eval_perplexity.item()}")
