import json
import torch
from torch.utils.data import IterableDataset


class StreamDataset(IterableDataset):
    def __init__(self, file_path, vocab, seq_len=256):
        self.file_path = file_path
        self.vocab = vocab
        self.seq_len = seq_len

    def __iter__(self):
        while True:  # Infinite loop for continuous training
            with open(self.file_path, 'r', encoding='utf-8') as f:
                buffer = []
                for line in f:
                    try:
                        data = json.loads(line)
                        text = f"{data.get('title', '')}{data.get('text', '')}{data.get('answer', '')}"
                        encoded = [self.vocab.get(c, 0) for c in text]
                        buffer.extend(encoded)

                        while len(buffer) > self.seq_len + 1:
                            yield (
                                torch.LongTensor(buffer[:self.seq_len]),
                                torch.LongTensor(buffer[1:self.seq_len + 1])
                            )
                            buffer = buffer[self.seq_len:]
                    except Exception:
                        continue  # Skip malformed lines