"""Fetch only the tokenizer and quantized ONNX weights."""

import hashlib
import json
import platform
import urllib.request
from pathlib import Path

from note_processing.encoder import MODEL_DIR

REPOSITORY = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
CHECKSUMS = {
    "tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
    "model_quint8_avx2.onnx": "b941bf19f1f1283680f449fa6a7336bb5600bdcd5f84d10ddc5cd72218a0fd21",
    "model_qint8_arm64.onnx": "4278337fd0ff3c68bfb6291042cad8ab363e1d9fbc43dcb499fe91c871902474",
}


def download_model(directory=MODEL_DIR):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    architecture = platform.machine().lower()
    variant = "model_qint8_arm64.onnx" if architecture in {"arm64", "aarch64"} else "model_quint8_avx2.onnx"
    files = {"tokenizer.json": "tokenizer.json", "model.onnx": f"onnx/{variant}"}
    hashes = {}
    for local_name, remote_name in files.items():
        target = directory / local_name
        if not target.exists():
            url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{remote_name}?download=true"
            temporary = target.with_suffix(target.suffix + ".part")
            print(f"Downloading {remote_name}…", flush=True)
            try:
                with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        hashes[local_name] = hashlib.sha256(target.read_bytes()).hexdigest()
        if hashes[local_name] != CHECKSUMS[remote_name.split("/")[-1]]:
            target.unlink()
            raise ValueError(f"Checksum mismatch for {remote_name}; run download again")
    manifest = {"repository": REPOSITORY, "revision": REVISION, "variant": variant,
                "license": "Apache-2.0", "sha256": hashes}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Model ready: {directory}")
