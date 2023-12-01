import pickle
from midi_processor.processor import decode_midi, encode_midi

with open("dataset/processed/alb_esp1_format0.mid.pickle", "rb") as f:
    data = pickle.load(f)
print(data)

decode_midi(data, file_path="generated.mid")
