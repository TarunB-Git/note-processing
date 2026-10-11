"""CPU sentence embeddings and contextual word vectors from quantized MiniLM."""

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from note_processing.paths import MODEL_DIR


@dataclass
class EncodedText:
    vector: np.ndarray
    words: list
    truncated: bool


def unit_vector(vector):
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


class Encoder:
    def __init__(self, model_dir=MODEL_DIR, threads=2):
        directory = Path(model_dir)
        self.directory = directory
        if not (directory / "model.onnx").exists() or not (directory / "tokenizer.json").exists():
            raise FileNotFoundError("Model missing. Run: python -m note_processing download")
        options = ort.SessionOptions()
        ort.disable_telemetry_events()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(directory / "model.onnx"), options,
                                            providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self.cache = OrderedDict()
        self.lock = RLock()
        self.input_names = {item.name for item in self.session.get_inputs()}

    def encode(self, texts):
        texts = list(texts)
        with self.lock:
            missing = list(dict.fromkeys(text for text in texts if text not in self.cache))
            # Keep requested vectors even when a large import exceeds the LRU cache.
            requested = {text: self.cache[text] for text in texts if text in self.cache}
            for offset in range(0, len(missing), 16):
                batch = missing[offset:offset + 16]
                encodings = self.tokenizer.encode_batch(batch)
                inputs = {
                    "input_ids": np.array([item.ids for item in encodings], dtype=np.int64),
                    "attention_mask": np.array([item.attention_mask for item in encodings], dtype=np.int64),
                    "token_type_ids": np.array([item.type_ids for item in encodings], dtype=np.int64),
                }
                tokens = self.session.run(None, {k: v for k, v in inputs.items() if k in self.input_names})[0]
                mask = inputs["attention_mask"][..., None]
                pooled = (tokens * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
                for text, encoding, vectors, vector in zip(batch, encodings, tokens, pooled):
                    groups = {}
                    for index, word_id in enumerate(encoding.word_ids):
                        if word_id is None or encoding.special_tokens_mask[index]:
                            continue
                        groups.setdefault(word_id, []).append(index)
                    words = []
                    for indexes in list(groups.values())[:64]:
                        start = min(encoding.offsets[i][0] for i in indexes)
                        end = max(encoding.offsets[i][1] for i in indexes)
                        words.append((start, end, unit_vector(vectors[indexes].mean(axis=0))))
                    encoded = EncodedText(unit_vector(vector).astype(np.float32), words,
                                          bool(encoding.overflowing))
                    self.cache[text] = encoded
                    requested[text] = encoded
                    if len(self.cache) > 128:
                        self.cache.popitem(last=False)
            for text in texts:
                if text in self.cache:
                    self.cache.move_to_end(text)
            return [requested[text] for text in texts]
