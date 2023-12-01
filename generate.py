import custom
from custom import criterion
from custom.layers import *
from custom.config import config
from model import MusicTransformer
from data import Data
import utils
from midi_processor.processor import decode_midi, encode_midi

import torch

import datetime
import argparse

# from tensorboardX import SummaryWriter


parser = custom.get_argument_parser()
args = parser.parse_args()
config.load(args.model_dir, args.configs, initialize=True)

# check cuda
if torch.cuda.is_available():
    config.device = torch.device("cuda")
else:
    config.device = torch.device("cpu")


# current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
# gen_log_dir = 'logs/mt_decoder/generate_'+current_time+'/generate'
# gen_summary_writer = SummaryWriter(gen_log_dir)

mt = torch.load("config/final.pth").to(config.device)
mt.test()

print(config.condition_file)
if config.condition_file is not None:
    inputs = np.array([encode_midi("dataset/midi/BENABD10.mid")[:500]])
else:
    inputs = np.array([[24, 28, 31]])
inputs = torch.from_numpy(inputs).to(config.device)
result = mt(inputs, config.length)

# torch.save(result, "generated.pt")

decode_midi(result, file_path="generated.mid")

# gen_summary_writer.close()
