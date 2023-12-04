import pickle
from midi_processor.processor import decode_midi, encode_midi
import torch

with open(
    "test_decode/2018/MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.midi.pickle",
    "rb",
) as f:
    data = pickle.load(f)

decode_midi(data, file_path="test.mid")
