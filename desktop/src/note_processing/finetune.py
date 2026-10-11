"""Optional SetFit encoder training, quantized export, and held-out comparison."""

import importlib.util
import json
import os
import selectors
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from note_processing.paths import APP_HOME


def start_finetune(workspace, progress):
    if not all(importlib.util.find_spec(name) for name in ('setfit', 'torch', 'onnx')):
        raise ValueError('Encoder training needs the optional dependencies. Run: pip install -e ".[training]". Head training works with the default install.')
    from note_processing.learning import split_labels
    records = workspace.store.corrections()
    split_labels(records, workspace.classifier.names, APP_HOME / 'training/split.json')
    APP_HOME.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=APP_HOME, prefix='training-') as temporary:
        source = Path(temporary) / 'labels.json'
        source.write_text(json.dumps({'records': records, 'classes': workspace.classifier.classes}))
        environment = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'TOKENIZERS_PARALLELISM': 'false'}
        process = subprocess.Popen([sys.executable, '-m', 'note_processing.finetune', str(source)],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment)
        try:
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            pending, percent, message = '', 0, 'Starting encoder training'
            log_path = APP_HOME / 'training/finetune.log'
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open('wb') as log:
                while process.poll() is None:
                    progress(percent, message)
                    for key, _ in selector.select(timeout=0.5):
                        chunk = os.read(key.fileobj.fileno(), 8192)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        log.write(chunk)
                        pending += chunk.decode('utf8', errors='replace').replace('\r', '\n')
                        while '\n' in pending:
                            line, pending = pending.split('\n', 1)
                            if line.startswith('PROGRESS '):
                                event = json.loads(line[9:])
                                percent, message = event['percent'], event['message']
            selector.close()
            if process.wait() != 0:
                raise ValueError('Encoder training failed. Details are in the training log.')
        except BaseException:
            process.terminate()
            process.wait()
            raise
    active = APP_HOME / 'active-encoder.json'
    if active.exists():
        from note_processing.encoder import Encoder
        with workspace.classifier.lock:
            workspace.classifier.encoder = Encoder(json.loads(active.read_text())['directory'])
            workspace.classifier.set_classes(workspace.classifier.classes)
            workspace.classifier.load_head()
            workspace.classifier.set_corrections(records)
    return json.loads((APP_HOME / 'training/report.json').read_text())


def export_encoder(body, directory):
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic

    directory = Path(directory)
    class TokenEncoder(torch.nn.Module):
        def __init__(self, transformer):
            super().__init__()
            self.transformer = transformer

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.transformer(input_ids=input_ids, attention_mask=attention_mask,
                                    token_type_ids=token_type_ids).last_hidden_state

    transformer = body[0].auto_model.cpu().eval()
    if hasattr(transformer, 'set_attn_implementation'):
        transformer.set_attn_implementation('eager')
    else:
        transformer.config._attn_implementation = 'eager'
    tokenizer = body.tokenizer
    # Padding in the tracing example keeps the attention mask in the exported graph.
    dummy = tokenizer(['Short.', 'An example note with several words.'], padding=True, return_tensors='pt')
    ids, mask = dummy['input_ids'], dummy['attention_mask']
    types = dummy.get('token_type_ids', torch.zeros_like(ids))
    full = directory / 'full.onnx'
    torch.onnx.export(TokenEncoder(transformer), (ids, mask, types), str(full),
        input_names=['input_ids', 'attention_mask', 'token_type_ids'], output_names=['last_hidden_state'],
        dynamic_axes={key: {0: 'batch', 1: 'sequence'} for key in ['input_ids', 'attention_mask', 'token_type_ids', 'last_hidden_state']},
        opset_version=17, dynamo=False)
    quantize_dynamic(str(full), str(directory / 'model.onnx'), weight_type=QuantType.QUInt8,
                     per_channel=True, extra_options={'MatMulConstBOnly': True})
    tokenizer.backend_tokenizer.save(str(directory / 'tokenizer.json'))
    full.unlink()


def finetune(source):
    import numpy as np
    import torch
    from datasets import Dataset
    from huggingface_hub import snapshot_download
    from setfit import SetFitModel, Trainer, TrainingArguments

    from note_processing.classifier import Classifier
    from note_processing.download import REPOSITORY, REVISION
    from note_processing.encoder import Encoder
    from note_processing.learning import classes_key, encoder_key, fit_head, matrices, metrics, softmax, split_labels
    from note_processing.signals import normalized

    def progress(percent, message):
        print('PROGRESS ' + json.dumps({'percent': percent, 'message': message}), flush=True)

    data = json.loads(Path(source).read_text())
    names = [c['name'] for c in data['classes']]
    train, test = split_labels(data['records'], names, APP_HOME / 'training/split.json')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    directory = APP_HOME / 'models' / ('personal-' + stamp)
    directory.mkdir(parents=True)
    progress(5, 'Loading the training model; first use downloads full precision weights')
    base = snapshot_download(REPOSITORY, revision=REVISION,
        allow_patterns=['*.json', '*.txt', '*.safetensors', '1_Pooling/*'])
    model = SetFitModel.from_pretrained(base)
    train_dataset = Dataset.from_list([{'text': r['text'], 'label': names.index(r['label'])} for r in train])
    args = TrainingArguments(batch_size=16, num_epochs=1, num_iterations=5, seed=42,
                             output_dir=str(directory / 'checkpoints'), report_to='none')
    progress(15, 'Fine-tuning sentence representations on confirmed notes')
    trainer = Trainer(model=model, args=args, train_dataset=train_dataset)
    trainer.train()
    model.model_body.save_pretrained(str(directory / 'sentence-transformer'))
    progress(60, 'Exporting the encoder to INT8 ONNX')

    export_encoder(model.model_body, directory)
    encoder = Encoder(directory)
    candidate = Classifier(data['classes'], encoder=encoder, load_head=False)
    held_text = {normalized(r['text']) for r in test}
    seeds = [r for r in candidate.seed_records() if normalized(r['text']) not in held_text]
    evaluation_classes = [{**c, 'examples': [s for s in c['examples'] if normalized(s) not in held_text]}
                          for c in data['classes']]
    x, y = matrices(encoder, seeds + train, names)
    weights, bias = fit_head(x, y, names)
    candidate.set_classes(evaluation_classes)
    candidate.install_head(weights, bias)
    # Evaluate the deployed rules + trained head, without held-out correction memory.
    candidate_predictions = [r['label'] for r in candidate.classify(test)]
    from note_processing.paths import MODEL_DIR
    baseline = Classifier(evaluation_classes, model_dir=MODEL_DIR, corrections=train, load_head=False)
    baseline_predictions = [r['label'] for r in baseline.classify(test)]
    report = {'kind': 'encoder', 'created': datetime.now(timezone.utc).isoformat(), 'train_count': len(train),
              'baseline_kind': 'starter with training-split correction memory',
              'test_count': len(test), 'candidate': metrics([r['label'] for r in test], candidate_predictions, names),
              'baseline': metrics([r['label'] for r in test], baseline_predictions, names),
              'mistakes': [{'text': r['text'], 'expected': r['label'], 'predicted': p}
                           for r, p in zip(test, candidate_predictions) if r['label'] != p]}
    report['adopted'] = report['candidate']['macro_f1'] >= report['baseline']['macro_f1']
    training = APP_HOME / 'training'
    training.mkdir(parents=True, exist_ok=True)
    if report['adopted']:
        progress(85, 'Fitting the personal head and saving the quantized model')
        x, y = matrices(encoder, seeds + train, names)
        weights, bias = fit_head(x, y, names)
        temporary = training / 'head.tmp.npz'
        np.savez(temporary, weights=weights, bias=bias, names=np.array(names), encoder_key=np.array(encoder_key(encoder)),
                 classes_key=np.array(classes_key(data['classes'])))
        temporary.replace(training / 'head.npz')
        active = APP_HOME / 'active-encoder.json'
        temporary = active.with_suffix('.tmp')
        temporary.write_text(json.dumps({'directory': str(directory)}))
        temporary.replace(active)
    (training / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    progress(100, 'Fine-tuning complete' if report['adopted'] else 'Candidate scored lower; baseline retained')


if __name__ == '__main__':
    import traceback
    try:
        finetune(sys.argv[1])
    except Exception:
        log = APP_HOME / 'training/finetune-error.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(traceback.format_exc())
        raise
