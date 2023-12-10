import pickle
from midi_processor.processor import decode_midi, encode_midi
import torch
import random

# with open(
#     "test_decode/2018/MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.midi.pickle",
#     "rb",
# ) as f:
#     data = pickle.load(f)

# data = []
# for _ in range(100_000):
#     data.append(random.randint(0, 387))

# decode_midi(data, file_path="test.mid")

result = torch.load("dataset/4-layers-probe.pth", map_location=torch.device("cpu"))
print(result["eval_y"][:, 2].mean())
