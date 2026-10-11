"""Local HTTP interface for the persistent note workspace."""

import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from note_processing.classifier import Classifier
from note_processing.deeper import local_models
from note_processing.outputs import output_zip
from note_processing.paths import PACKAGE_DIR
from note_processing.workspace import Workspace

MAX_BODY_BYTES = 32 * 1024 * 1024


def create_server(classifier, root, port=8765, output=None, store=None):
    workspace = Workspace(classifier, root, output or Path.cwd() / 'organized', store)
    token = secrets.token_urlsafe(24)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def trusted_request(self):
            hosts = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
            if self.headers.get('Host') not in hosts or self.headers.get('Origin') and self.headers['Origin'] not in {'http://' + h for h in hosts}:
                self.send_json({'error': 'Invalid host or origin'}, 403)
                return False
            return True

        def authorized(self):
            if self.headers.get('X-Note-Token') != token:
                self.send_json({'error': 'Missing session token'}, 403)
                return False
            return True

        def send_json(self, value, status=200):
            self.send_data(json.dumps(value, ensure_ascii=False).encode(), 'application/json; charset=utf-8', status)

        def send_data(self, data, content_type, status=200, attachment=None):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            if attachment:
                self.send_header('Content-Disposition', f'attachment; filename="{attachment}"')
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if not self.trusted_request():
                return
            route = urlsplit(self.path)
            try:
                if route.path == '/api/config':
                    self.send_json({'token': token, 'root': str(workspace.root), 'output_root': str(workspace.output)})
                elif route.path.startswith('/api/'):
                    if not self.authorized():
                        return
                    if route.path == '/api/workspace':
                        self.send_json(workspace.snapshot())
                    elif route.path == '/api/outputs':
                        self.send_json({'outputs': workspace.outputs()})
                    elif route.path == '/api/models':
                        self.send_json({'models': local_models()})
                    elif route.path.startswith('/api/jobs/'):
                        self.send_json(workspace.jobs.get(route.path.rsplit('/', 1)[1]))
                    elif route.path == '/api/folders':
                        value = parse_qs(route.query).get('path', [''])[0]
                        path = workspace.selected_path(value)
                        entries = [{'name': child.name, 'path': child.relative_to(workspace.root).as_posix(), 'directory': child.is_dir()}
                                   for child in sorted(path.iterdir()) if not child.is_symlink() and not child.name.startswith('.')
                                   and (child.is_dir() or child.suffix.lower() in {'.txt', '.md', '.markdown'})]
                        self.send_json({'path': path.relative_to(workspace.root).as_posix(), 'entries': entries[:500]})
                    else:
                        self.send_json({'error': 'Not found'}, 404)
                else:
                    files = {'/': ('index.html', 'text/html; charset=utf-8'), '/app.js': ('app.js', 'text/javascript; charset=utf-8'), '/style.css': ('style.css', 'text/css; charset=utf-8')}
                    if route.path not in files:
                        self.send_json({'error': 'Not found'}, 404)
                        return
                    filename, mime = files[route.path]
                    self.send_data((PACKAGE_DIR / 'web' / filename).read_bytes(), mime)
            except (OSError, ValueError) as error:
                self.send_json({'error': str(error)}, 400)

        def do_POST(self):
            if not self.trusted_request() or not self.authorized():
                return
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                self.send_json({'error': 'Expected JSON'}, 415)
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= MAX_BODY_BYTES:
                    raise ValueError('Request must be between 1 byte and 32 MB')
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError('Expected an object')
                route = urlsplit(self.path).path
                if route == '/api/import':
                    self.send_json({'job': workspace.jobs.start(lambda progress: workspace.import_notes(data, progress))})
                elif route == '/api/reclassify':
                    self.send_json({'job': workspace.jobs.start(lambda progress: workspace.reclassify(data, progress))})
                elif route == '/api/train':
                    if data.get('kind', 'head') == 'encoder':
                        from note_processing.finetune import start_finetune
                        self.send_json({'job': workspace.jobs.start(lambda progress: start_finetune(workspace, progress))})
                    else:
                        self.send_json({'job': workspace.jobs.start(workspace.train)})
                elif route == '/api/cancel':
                    workspace.jobs.cancel(data.get('job'))
                    self.send_json({'ok': True})
                elif route == '/api/move':
                    self.send_json(workspace.move(data))
                elif route == '/api/group':
                    self.send_json(workspace.group(data))
                elif route == '/api/undo':
                    workspace.store.undo_move()
                    classifier.set_corrections(workspace.store.corrections())
                    self.send_json({'ok': True})
                elif route == '/api/update':
                    self.send_json({'notes': workspace.update_note(data)})
                elif route == '/api/classes':
                    names = [c.get('name') for c in data.get('classes', []) if isinstance(c, dict)]
                    used = {n['label'] for n in workspace.store.notes()}
                    if not used.issubset(names):
                        raise ValueError('Move notes out of a category before removing it')
                    classifier.save_classes(data.get('classes'))
                    self.send_json({'classes': classifier.classes})
                elif route == '/api/output-settings':
                    workspace.configure_outputs(data.get('targets'))
                    self.send_json({'ok': True})
                elif route == '/api/save-outputs':
                    self.send_json(workspace.save_outputs(data.get('overwrite') is True))
                elif route == '/api/download':
                    self.send_data(output_zip(workspace.outputs()), 'application/zip', attachment='organized-notes.zip')
                elif route == '/api/training-data':
                    content = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in workspace.store.corrections())
                    self.send_data(content.encode(), 'application/x-ndjson', attachment='confirmed-notes.jsonl')
                else:
                    self.send_json({'error': 'Not found'}, 404)
            except (OSError, ValueError, TypeError, KeyError) as error:
                self.send_json({'error': str(error)}, 400)

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.workspace = workspace
    return server


def serve(root, port=8765, classes=None, output=None):
    server = create_server(Classifier(classes), root, port, output)
    print(f'Note Processing: http://127.0.0.1:{server.server_port}', flush=True)
    print(f'Input: {server.workspace.root}\nOutput: {server.workspace.output}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.workspace.jobs.close()
