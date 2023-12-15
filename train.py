from model import MusicTransformer
import custom
from custom.metrics import *
from custom.criterion import SmoothCrossEntropyLoss, CustomSchedule
from custom.config import config
from data import Data
import torch
import torch.optim as optim
from tqdm import tqdm


# set config
parser = custom.get_argument_parser()
args = parser.parse_args()
config.load(args.model_dir, args.configs, initialize=True)
if torch.cuda.is_available():
    config.device = torch.device("cuda")
    print("\nUsing CUDA\n")
else:
    config.device = torch.device("cpu")
    print("\nUsing CPU\n")
print(config)
learning_rate = config.l_r


# load data
dataset = Data(config.pickle_dir)
print(dataset)

# define model
mt = MusicTransformer(
    embedding_dim=config.embedding_dim,
    vocab_size=config.vocab_size,
    num_layer=config.num_layers,
    max_seq=config.max_seq,
    dropout=config.dropout,
    debug=config.debug,
    loader_path=config.load_path,
)
mt.to(config.device)
opt = optim.Adam(mt.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
scheduler = CustomSchedule(config.embedding_dim, optimizer=opt)

# init metric set
metric_set = MetricsSet(
    {
        "accuracy": CategoricalAccuracy(),
        "loss": SmoothCrossEntropyLoss(
            config.label_smooth, config.vocab_size, config.pad_token
        ),
        "bucket": LogitsBucketting(config.vocab_size),
    }
)


# Summary
print(mt)
print("| Summary - Device Info : {}".format(torch.cuda.device))


# Train Start
print(">> Train start...")
idx = 0
for e in range(config.epochs):
    print(">>> [Epoch was updated]")
    for b in tqdm(range(len(dataset.files) // config.batch_size)):
        # Update Model
        scheduler.optimizer.zero_grad()
        try:
            batch_x, batch_y = dataset.slide_seq2seq_batch(
                config.batch_size, config.max_seq
            )
            batch_x = (
                torch.from_numpy(batch_x)
                .contiguous()
                .to(config.device, non_blocking=True, dtype=torch.int)
            )
            batch_y = (
                torch.from_numpy(batch_y)
                .contiguous()
                .to(config.device, non_blocking=True, dtype=torch.int)
            )
        except IndexError:
            continue
        mt.train()
        sample = mt.forward(batch_x)
        metrics = metric_set(sample, batch_y)
        loss = metrics["loss"]
        loss.backward()
        scheduler.step()

        # Evaluate at the end of each epoch
        if b == len(dataset.files) // config.batch_size - 1:
            mt.eval()
            eval_x, eval_y = dataset.slide_seq2seq_batch(
                config.batch_size, config.max_seq, "eval"
            )
            eval_x = (
                torch.from_numpy(eval_x).contiguous().to(config.device, dtype=torch.int)
            )
            eval_y = (
                torch.from_numpy(eval_y).contiguous().to(config.device, dtype=torch.int)
            )
            eval_preiction, weights = mt.forward(eval_x)
            eval_metrics = metric_set(eval_preiction, eval_y)
            print("\n====================================================")
            print("Epoch: {}/{}".format(e + 1, config.epochs))
            print(
                "Train >>>> Loss: {:6.6}, Accuracy: {}".format(
                    metrics["loss"], metrics["accuracy"]
                )
            )
            print(
                "Eval >>>> Loss: {:6.6}, Accuracy: {}".format(
                    eval_metrics["loss"], eval_metrics["accuracy"]
                )
            )

        torch.cuda.empty_cache()
        idx += 1

# Save Model
torch.save(mt, "trained_models/final.pth".format(idx))
