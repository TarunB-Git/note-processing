"""Run the local note organizer or sort a file from the terminal."""

import argparse
import json
from datetime import date
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    download = commands.add_parser('download', help='Download the small quantized encoder')
    download.add_argument('--directory', type=Path)
    app = commands.add_parser('serve', help='Open the persistent local workspace')
    app.add_argument('--root', type=Path, default=Path.cwd())
    app.add_argument('--output', type=Path, default=Path.cwd() / 'organized')
    app.add_argument('--port', type=int, default=8765)
    app.add_argument('--classes', type=Path)
    organize = commands.add_parser('organize', help='Split notes into category files')
    classify = commands.add_parser('classify', help='Print classifications as JSON')
    for command in (organize, classify):
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument('--text')
        source.add_argument('--file', type=Path)
        source.add_argument('--folder', type=Path)
        command.add_argument('--mode', choices=['lines', 'paragraphs', 'sentences'], default='lines')
        command.add_argument('--classes', type=Path)
        command.add_argument('--anchor', type=date.fromisoformat)
    organize.add_argument('--output', type=Path, required=True)
    organize.add_argument('--overwrite', action='store_true')
    classify.add_argument('--out', type=Path)
    train = commands.add_parser('train', help='Train on locally confirmed notes')
    train.add_argument('--encoder', action='store_true', help='Fine-tune MiniLM with the optional training dependencies')
    args = parser.parse_args()
    try:
        if args.command == 'download':
            from note_processing.download import download_model
            download_model(args.directory) if args.directory else download_model()
            return
        from note_processing.classifier import Classifier, load_classes
        if args.command == 'serve':
            from note_processing.server import serve
            serve(args.root, args.port, load_classes(args.classes), args.output)
            return
        from note_processing.store import Store
        store = Store()
        classifier = Classifier(load_classes(getattr(args, 'classes', None)), corrections=store.corrections())
        if args.command == 'train':
            from note_processing.workspace import Workspace
            workspace = Workspace(classifier, Path.cwd(), Path.cwd() / 'organized', store)
            if args.encoder:
                from note_processing.finetune import start_finetune
                report = start_finetune(workspace, lambda percent, message: print(f'{percent}% {message}', flush=True))
            else:
                report = workspace.train(lambda percent, message: print(f'{percent}% {message}', flush=True))
            print(json.dumps(report, indent=2))
            workspace.jobs.close()
            return
        from note_processing.documents import read_file, scan_folder, split_document
        warnings, protected = [], []
        if args.text is not None:
            units = split_document(args.text, mode=args.mode)
        elif args.file:
            units = read_file(args.file, mode=args.mode)
            protected = [args.file]
        else:
            scan = scan_folder(args.folder, args.mode)
            units, warnings = scan['units'], scan['warnings']
            protected = [args.folder / name for name in scan['files']]
        results = classifier.classify(units, args.anchor)
        if args.command == 'organize':
            from note_processing.outputs import render_outputs, write_outputs
            rows = render_outputs(classifier.classes, results)
            write_outputs(args.output, rows, overwrite=args.overwrite, protected=protected)
            for row in rows:
                print(f'{row["count"]:>5}  {args.output / row["path"]}')
            for warning in warnings:
                print('Warning: ' + warning)
        else:
            content = json.dumps({'results': results, 'warnings': warnings}, indent=2, ensure_ascii=False) + '\n'
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(content, encoding='utf8')
            else:
                print(content, end='')
    except (OSError, ValueError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
