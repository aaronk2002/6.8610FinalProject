from model import MusicTransformer
import custom
from custom.metrics import *
from custom.criterion import SmoothCrossEntropyLoss, CustomSchedule
from custom.config import config
from data import Data

import utils
import datetime
import time

import torch
import torch.optim as optim

from tqdm import tqdm

# from tensorboardX import SummaryWriter


# set config
parser = custom.get_argument_parser()
args = parser.parse_args()
config.load(args.model_dir, args.configs, initialize=True)

# check cuda
if torch.cuda.is_available():
    config.device = torch.device("cuda")
    print("\nUsing CUDA\n")
else:
    config.device = torch.device("cpu")
    print("\nUsing CPU\n")

dataset = Data(config.pickle_dir)
print(dataset)

metric_set = MetricsSet(
    {
        "accuracy": CategoricalAccuracy(),
        "loss": SmoothCrossEntropyLoss(
            config.label_smooth, config.vocab_size, config.pad_token
        ),
        "bucket": LogitsBucketting(config.vocab_size),
    }
)

print("\n\nNew mt\n")
new_mt = torch.load(args.model_dir + "/final.pth")
new_mt.eval()
eval_x, eval_y = dataset.slide_seq2seq_batch(config.batch_size, config.max_seq, "eval")
eval_x = torch.from_numpy(eval_x).contiguous().to(config.device, dtype=torch.int)
eval_y = torch.from_numpy(eval_y).contiguous().to(config.device, dtype=torch.int)

eval_preiction, weights = new_mt.forward(eval_x)

eval_metrics = metric_set(eval_preiction, eval_y)
print(
    "Eval >>>> Loss: {:6.6}, Accuracy: {}".format(
        eval_metrics["loss"], eval_metrics["accuracy"]
    )
)
