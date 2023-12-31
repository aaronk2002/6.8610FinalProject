import custom
from custom.layers import *
from custom.config import config
from midi_processor.processor import decode_midi, encode_midi
import torch


# Get configs
parser = custom.get_argument_parser()
args = parser.parse_args()
config.load(args.model_dir, args.configs, initialize=True)
if torch.cuda.is_available():
    config.device = torch.device("cuda")
else:
    config.device = torch.device("cpu")
print(config.condition_file)


# Load Model
mt = torch.load("trained_models/final.pth").to(config.device)
mt.test()

# Load initialization
if config.condition_file is not None:
    inputs = np.array([encode_midi("dataset/midi/BENABD10.mid")[:500]])
else:
    initialize = torch.load("init_generate.pt")
    inputs = np.array([initialize])
inputs = torch.from_numpy(inputs).to(config.device)
result = mt(inputs, config.length)

decode_midi(result, file_path="generated.mid")
