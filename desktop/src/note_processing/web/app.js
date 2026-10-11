const $ = id => document.getElementById(id);
const colors = ['#8295b3', '#ac8acf', '#739b83', '#d1a062', '#679cbb', '#ac8c91'];
const state = {
  token: '', notes: [], classes: [], view: 'notes', filter: 'all', layout: 'board',
  selected: new Set(), documents: [], input: 'paste', page: 0, limits: {}, outputs: [], output: null,
  learning: {}, job: null, poll: null, searchTimer: null
};

function node(tag, className, text) {
  const value = document.createElement(tag);
  if (className) value.className = className;
  if (text !== undefined) value.textContent = text;
  return value;
}

async function request(path, body, binary = false) {
  const response = await fetch(path, {
    method: body === undefined ? 'GET' : 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Note-Token': state.token },
    ...(body === undefined ? {} : { body: JSON.stringify(body) })
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Request failed');
  }
  return binary ? response.blob() : response.json();
}

function showError(message) {
  $('error').textContent = message;
  $('error').hidden = !message;
  document.querySelectorAll('.dialog-error').forEach(el => el.remove());
  const dialog = document.querySelector('dialog[open]');
  if (message && dialog) {
    const error = node('div', 'error dialog-error', message);
    error.setAttribute('role', 'alert');
    dialog.querySelector('.dialog-heading').after(error);
  }
}

async function run(operation) {
  showError('');
  try { await operation(); } catch (error) { showError(error.message); }
}

function toast(message) {
  $('toast').textContent = message;
  $('toast').hidden = false;
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => { $('toast').hidden = true; }, 4500);
}

function title(name) { return state.classes.find(c => c.name === name)?.title || name; }
function color(name) { return colors[Math.max(0, state.classes.findIndex(c => c.name === name)) % colors.length]; }
function categorySelect(value, label) {
  const select = node('select');
  select.setAttribute('aria-label', label);
  state.classes.forEach(category => {
    const option = node('option', '', category.title); option.value = category.name;
    option.selected = category.name === value; select.append(option);
  });
  return select;
}

async function refresh() {
  const workspace = await request('/api/workspace');
  state.notes = workspace.notes;
  state.classes = workspace.classes;
  state.learning = workspace.learning;
  state.targets = workspace.targets;
  state.context = workspace.context;
  state.neighborWords = workspace.neighbor_words;
  $('input-root').textContent = workspace.root;
  $('output-root').textContent = workspace.output_root;
  $('model-status').textContent = workspace.learning.head === 'personal' ? 'MiniLM · personal classifier' : 'MiniLM · starter classifier';
  state.selected = new Set([...state.selected].filter(id => state.notes.some(n => n.id === id)));
  render();
  if (state.view === 'outputs') await loadOutputs();
}

function visibleNotes() {
  const query = $('search').value.trim().toLowerCase();
  return state.notes.filter(note =>
    (state.filter === 'all' || (state.filter === 'review' ? note.needs_review : note.label === state.filter)) &&
    (!query || (note.text + ' ' + note.source).toLowerCase().includes(query))
  );
}

function renderSidebar() {
  $('nav-total').textContent = state.notes.length;
  $('review-count').textContent = state.notes.filter(n => n.needs_review).length;
  $('review-filter').classList.toggle('active', state.filter === 'review' && state.view === 'notes');
  const nav = $('categories'); nav.replaceChildren();
  state.classes.forEach(category => {
    const button = node('button', 'nav-button' + (state.filter === category.name && state.view === 'notes' ? ' active' : ''));
    const dot = node('span', 'category-dot'); dot.style.setProperty('--category-color', color(category.name));
    button.append(dot, node('span', 'category-name', category.title), node('span', '', state.notes.filter(n => n.label === category.name).length));
    button.onclick = () => { state.filter = category.name; state.page = 0; setView('notes'); };
    button.ondragover = event => { if (event.dataTransfer.types.includes('application/x-note-ids')) { event.preventDefault(); button.classList.add('active'); } };
    button.ondragleave = () => button.classList.toggle('active', state.filter === category.name);
    button.ondrop = event => dropNotes(event, category.name);
    nav.append(button);
  });
}

function noteText(note) {
  const paragraph = node('p', 'note-text');
  const characters = Array.from(note.text);
  let cursor = 0;
  for (const word of ($('highlights').checked ? note.highlights || [] : [])) {
    if (word.start < cursor) continue;
    paragraph.append(document.createTextNode(characters.slice(cursor, word.start).join('')), node('mark', '', characters.slice(word.start, word.end).join('')));
    cursor = word.end;
  }
  paragraph.append(document.createTextNode(characters.slice(cursor).join('')));
  return paragraph;
}

function selectionBox(note) {
  const input = node('input'); input.type = 'checkbox'; input.checked = state.selected.has(note.id);
  input.setAttribute('aria-label', 'Select note: ' + note.text.slice(0, 80));
  input.onclick = event => event.stopPropagation();
  input.onchange = () => { input.checked ? state.selected.add(note.id) : state.selected.delete(note.id); render(); };
  return input;
}

function card(note) {
  const article = node('article', 'note-card' + (note.kind === 'code' ? ' code-card' : '') + (state.selected.has(note.id) ? ' selected' : ''));
  article.draggable = true;
  article.dataset.noteId = note.id;
  article.tabIndex = 0;
  article.setAttribute('aria-label', title(note.label) + ': ' + note.text.slice(0, 120));
  const top = node('div', 'card-top');
  const more = node('button', '', '⋯'); more.setAttribute('aria-label', 'Details and move: ' + note.text.slice(0, 60));
  more.onclick = event => { event.stopPropagation(); showNote(note.id); };
  top.append(selectionBox(note), node('span', 'note-source', `${note.source.split('/').pop()} · ${note.line}`), more);
  const footer = node('div', 'card-footer');
  if (note.manual) footer.append(node('span', 'confirmed-tag', 'Confirmed'));
  else if (note.needs_review) footer.append(node('span', 'review-tag', 'Review'));
  if (note.date) footer.append(node('span', 'date-tag', note.date));
  if (note.label === 'todos') {
    const complete = node('button', '', note.done ? 'Completed' : 'Mark done');
    complete.onclick = event => { event.stopPropagation(); run(async () => { await request('/api/update', { ids: [note.id], done: !note.done }); await refresh(); }); };
    footer.append(complete);
  } else if (!note.manual) {
    const confirm = node('button', '', 'Confirm');
    confirm.onclick = event => { event.stopPropagation(); run(() => move([note.id], note.label)); };
    footer.append(confirm);
  }
  article.append(top, noteText(note), footer);
  article.onclick = () => showNote(note.id);
  article.onkeydown = event => { if (event.target === article && (event.key === 'Enter' || event.key === ' ')) { event.preventDefault(); showNote(note.id); } };
  article.ondragstart = event => {
    const ids = state.selected.has(note.id) ? [...state.selected] : [note.id];
    event.dataTransfer.setData('application/x-note-ids', JSON.stringify(ids));
    event.dataTransfer.effectAllowed = 'move';
  };
  return article;
}

function dropNotes(event, category) {
  if (!event.dataTransfer.types.includes('application/x-note-ids')) return;
  event.preventDefault(); event.stopPropagation();
  document.querySelectorAll('.dragover').forEach(el => el.classList.remove('dragover'));
  run(async () => {
    const ids = JSON.parse(event.dataTransfer.getData('application/x-note-ids'));
    if (!Array.isArray(ids)) throw new Error('Invalid note selection');
    await move(ids, category);
  });
}

function renderBoard(visible) {
  $('board').replaceChildren();
  const categories = state.filter !== 'all' && state.filter !== 'review' ? state.classes.filter(c => c.name === state.filter) : state.classes;
  categories.forEach(category => {
    const notes = visible.filter(n => n.label === category.name);
    const column = node('section', 'column'); column.setAttribute('aria-label', category.title + ' section');
    const heading = node('div', 'column-heading');
    const dot = node('span', 'category-dot'); dot.style.setProperty('--category-color', color(category.name));
    heading.append(dot, node('strong', '', category.title), node('span', 'count', notes.length));
    const list = node('div', 'column-list');
    const limit = state.limits[category.name] || 20;
    notes.slice(0, limit).forEach(note => list.append(card(note)));
    if (!notes.length) list.append(node('div', 'column-empty', 'Drop notes here'));
    column.append(heading, list);
    if (notes.length > limit) {
      const more = node('button', 'show-more', `Show ${Math.min(20, notes.length - limit)} more`);
      more.onclick = () => { state.limits[category.name] = limit + 20; render(); }; column.append(more);
    }
    column.ondragover = event => { if (event.dataTransfer.types.includes('application/x-note-ids')) { event.preventDefault(); column.classList.add('dragover'); } };
    column.ondragleave = event => { if (!column.contains(event.relatedTarget)) column.classList.remove('dragover'); };
    column.ondrop = event => dropNotes(event, category.name);
    $('board').append(column);
  });
}

function renderTable(visible) {
  const pages = Math.max(1, Math.ceil(visible.length / 100));
  state.page = Math.min(state.page, pages - 1);
  const notes = visible.slice(state.page * 100, (state.page + 1) * 100);
  $('note-table').replaceChildren();
  notes.forEach(note => {
    const row = node('tr'); const check = node('td'); check.append(selectionBox(note));
    const cell = node('td', 'note-cell'); const open = node('button', '', note.text.length > 300 ? note.text.slice(0, 300) + '…' : note.text);
    open.onclick = () => showNote(note.id); cell.append(open);
    const category = node('td'); const select = categorySelect(note.label, 'Category for ' + note.text.slice(0, 60));
    select.onchange = () => run(() => move([note.id], select.value)); category.append(select);
    row.append(check, cell, category, node('td', '', `${note.source} · ${note.line}`), node('td', '', note.date || '-'));
    $('note-table').append(row);
  });
  $('select-page').checked = notes.length > 0 && notes.every(n => state.selected.has(n.id));
  $('select-page').onchange = event => { notes.forEach(n => event.target.checked ? state.selected.add(n.id) : state.selected.delete(n.id)); render(); };
  $('page-info').textContent = `Page ${state.page + 1} of ${pages} · ${visible.length} notes`;
  $('previous-page').disabled = state.page === 0; $('next-page').disabled = state.page === pages - 1;
}

function render() {
  renderSidebar();
  const visible = visibleNotes();
  $('summary').textContent = `${visible.length} ${state.filter === 'all' ? 'notes' : title(state.filter).toLowerCase()}${visible.length !== state.notes.length ? ` / ${state.notes.length} total` : ''} · moves teach the classifier`;
  $('empty').hidden = state.notes.length > 0;
  $('board').hidden = !state.notes.length || state.layout !== 'board';
  $('table-wrap').hidden = !state.notes.length || state.layout !== 'table';
  $('bulk-bar').hidden = !state.selected.size;
  $('selection-count').textContent = state.selected.size + ' selected';
  const value = $('bulk-category').value || state.notes.find(n => state.selected.has(n.id))?.label || state.classes[0]?.name;
  const options = categorySelect(value, 'Move selected notes to category'); $('bulk-category').replaceChildren(...options.children);
  if (value) $('bulk-category').value = value;
  if (state.notes.length && state.layout === 'board') renderBoard(visible);
  if (state.notes.length && state.layout === 'table') renderTable(visible);
  renderLearning();
  document.querySelectorAll('[data-view]').forEach(button => button.classList.toggle('active', button.dataset.view === state.view));
  $('import').disabled = Boolean(state.job);
}

function setView(view) {
  state.view = view;
  ['notes', 'outputs', 'learning'].forEach(name => $(name + '-view').hidden = name !== view);
  $('page-title').textContent = { notes: 'Notes', outputs: 'Output files', learning: 'Learning' }[view];
  $('page-subtitle').textContent = { notes: 'Sort a file, correct a category, and keep the useful parts.', outputs: 'Preview the organized files and choose where each category belongs.', learning: 'Turn corrections into a classifier that follows your choices.' }[view];
  render();
  if (view === 'outputs') run(loadOutputs);
}

async function move(ids, label) {
  await request('/api/move', { ids, label });
  state.selected.clear();
  await refresh();
  toast(`${ids.length} ${ids.length === 1 ? 'note' : 'notes'} confirmed in ${title(label)}. Correction saved.`);
}

function showNote(identifier) {
  const note = state.notes.find(n => n.id === identifier);
  if (!note) return;
  $('detail-title').textContent = title(note.label);
  const detail = $('note-detail'); detail.replaceChildren();
  const text = node('div', 'detail-text'); text.append(noteText(note)); detail.append(text);
  detail.append(node('p', 'muted', `${note.source} · line ${note.line} · ${note.manual ? 'confirmed' : note.label_source.replaceAll('_', ' ')}`));
  detail.append(node('p', 'detail-reason', note.reason));
  if (note.evidence) detail.append(node('p', 'detail-quote', 'Evidence: ' + note.evidence));
  if (note.alternative) detail.append(node('p', 'muted', 'Alternative: ' + title(note.alternative)));
  (note.review_reasons || []).forEach(reason => detail.append(node('p', 'muted', reason)));
  const label = node('label', '', 'Category'); const select = categorySelect(note.label, 'Note category'); label.append(select); detail.append(label);
  const buttons = node('div', 'button-row'); const save = node('button', 'primary', 'Confirm category & teach');
  save.onclick = () => run(async () => { await move([identifier], select.value); $('note-dialog').close(); });
  const reclassify = node('button', 'secondary', 'Reclassify this note');
  reclassify.onclick = () => { state.selected = new Set([identifier]); $('note-dialog').close(); openReclassify(); render(); };
  buttons.append(save, reclassify); detail.append(buttons);
  if (note.label === 'todos') {
    const doneLabel = node('label', 'checkbox-label', 'Completed'); const check = node('input'); check.type = 'checkbox'; check.checked = Boolean(note.done);
    check.onchange = () => run(async () => { await request('/api/update', { ids: [identifier], done: check.checked }); await refresh(); });
    doneLabel.prepend(check); doneLabel.style.marginTop = '18px'; detail.append(doneLabel);
  }
  if (note.label_source !== 'local_model') {
    const list = node('div', 'score-list');
    list.append(node('p', 'eyebrow', 'CLASSIFIER SCORES'));
    Object.entries(note.scores || {}).sort((a, b) => b[1] - a[1]).forEach(([name, score]) => {
      const row = node('div', 'score-row'); const track = node('div', 'score-track'); const fill = node('div', 'score-fill');
      fill.style.width = `${Math.max(0, Math.min(1, score)) * 100}%`; track.append(fill);
      row.append(node('span', '', title(name)), track, node('span', '', score.toFixed(2))); list.append(row);
    });
    list.append(node('p', 'muted', 'Head probabilities are not calibrated confidence. Format cues and your corrections can override the head. Word matches show closeness to an example, not a causal explanation.'));
    detail.append(list);
  }
  if (note.matched_example) { detail.append(node('p', 'eyebrow', 'CLOSEST EXAMPLE'), node('p', 'detail-quote', note.matched_example)); }
  $('note-dialog').showModal();
}

async function startJob(path, body) {
  if (state.job) throw new Error('A job is already running');
  const response = await request(path, body);
  state.job = response.job;
  $('job-panel').hidden = false;
  $('job-message').textContent = 'Starting…'; $('job-progress').value = 0;
  render();
  async function poll() {
    try {
      const job = await request('/api/jobs/' + state.job);
      $('job-message').textContent = job.message; $('job-progress').value = job.percent;
      if (['complete', 'failed', 'cancelled'].includes(job.status)) {
        state.job = null;
        $('job-panel').hidden = true;
        await refresh();
        if (job.status === 'failed') showError(job.error);
        else if (job.status === 'cancelled') toast(job.message);
        else {
          const warnings = job.result?.warnings || [];
          toast(warnings.length ? 'Completed. ' + warnings.join('; ') : 'Completed. Workspace saved.');
        }
        return;
      }
      state.poll = setTimeout(poll, 900);
    } catch (error) { state.job = null; $('job-panel').hidden = true; showError(error.message); render(); }
  }
  await poll();
}

function openImport() {
  showError(''); $('import-context').value = state.context || '';
  $('import-neighbors').value = state.neighborWords || 0; $('import-dialog').showModal();
}
function setInput(kind) {
  state.input = kind;
  ['paste', 'files', 'local'].forEach(name => $(name + '-panel').hidden = name !== kind);
  document.querySelectorAll('[data-input]').forEach(button => { const active = button.dataset.input === kind; button.classList.toggle('active', active); button.setAttribute('aria-selected', active); });
}

async function importFiles(files) {
  const selected = [...files].filter(file => /\.(txt|md|markdown)$/i.test(file.name) && !file.webkitRelativePath.split('/').some(part => part.startsWith('.')));
  if (selected.length > 250) throw new Error('Choose up to 250 text or Markdown files');
  if (selected.some(file => file.size > 4 * 1024 * 1024)) throw new Error('Each file must be under 4 MB');
  if (selected.reduce((sum, file) => sum + file.size, 0) > 32 * 1024 * 1024) throw new Error('Files must total under 32 MB');
  state.documents = await Promise.all(selected.map(async file => ({ name: file.webkitRelativePath || file.name, text: await file.text() })));
  $('file-summary').textContent = `${state.documents.length} files selected`;
}

async function browse() {
  const response = await request('/api/folders?path=' + encodeURIComponent($('local-path').value));
  const list = $('folder-list'); list.replaceChildren();
  const parent = node('button', '', '↑ Parent folder'); parent.type = 'button';
  parent.onclick = () => { $('local-path').value = response.path.split('/').slice(0, -1).join('/'); run(browse); }; list.append(parent);
  response.entries.forEach(entry => {
    const button = node('button', '', (entry.directory ? '▸ ' : '· ') + entry.name); button.type = 'button';
    button.onclick = () => { $('local-path').value = entry.path; if (entry.directory) run(browse); }; list.append(button);
  });
}

async function openReclassify() {
  showError('');
  const count = state.selected.size || visibleNotes().length;
  $('reclassify-scope').textContent = `${count} ${state.selected.size ? 'selected' : 'visible'} notes. Confirmed notes stay pinned unless you include them below.`;
  $('reclassify-context').value = state.context || '';
  $('reclassify-neighbors').value = state.neighborWords || 0;
  $('reclassify-dialog').showModal();
  await run(async () => {
    const { models } = await request('/api/models');
    $('local-model').replaceChildren();
    models.forEach(model => { const option = node('option', '', model); option.value = model; option.selected = model === 'qwen3:1.7b'; $('local-model').append(option); });
    if (!models.length) { const option = node('option', '', 'No local model installed'); option.value = 'qwen3:1.7b'; $('local-model').append(option); }
  });
}

async function loadOutputs() {
  const response = await request('/api/outputs');
  state.outputs = response.outputs;
  state.output = state.outputs.some(row => row.category === state.output) ? state.output : state.outputs[0]?.category;
  const list = $('output-list'); list.replaceChildren();
  state.outputs.forEach(row => {
    const item = node('section', 'output-item' + (state.output === row.category ? ' active' : ''));
    const button = node('button'); button.append(node('strong', '', row.title), node('small', '', row.count + ' notes'));
    button.onclick = () => { state.output = row.category; document.querySelectorAll('.output-item').forEach(el => el.classList.remove('active')); item.classList.add('active'); renderPreview(); };
    const label = node('label', '', 'Output filename'); const input = node('input'); input.value = row.path; input.dataset.category = row.category; input.setAttribute('aria-label', row.title + ' output filename');
    input.onchange = () => run(async () => {
      const targets = outputTargets();
      await request('/api/output-settings', { targets }); await loadOutputs(); toast('Output filename saved.');
    });
    label.append(input); item.append(button, label); list.append(item);
  });
  renderPreview();
}

function renderPreview() {
  const row = state.outputs.find(item => item.category === state.output);
  $('preview-name').textContent = row?.path || 'Output preview';
  $('output-preview').textContent = row ? $('preview-kind').value === 'saved' ? row.saved_content ?? 'This file has not been saved yet.' : row.content : '';
}

function outputTargets() {
  return Object.fromEntries([...document.querySelectorAll('.output-item input')].map(input => [input.dataset.category, input.value]));
}

async function download(path, filename) {
  const blob = await request(path, {}, true);
  const url = URL.createObjectURL(blob); const link = node('a');
  link.href = url; link.download = filename; document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}

function renderLearning() {
  const learning = state.learning;
  $('confirmed-count').textContent = learning.confirmed || 0;
  $('active-head').textContent = learning.head === 'personal' ? 'Personal' : 'Starter';
  $('training-counts').replaceChildren();
  state.classes.forEach(category => {
    const count = learning.counts?.[category.name] || 0;
    const row = node('div', 'training-row'); const dot = node('span', 'category-dot'); dot.style.setProperty('--category-color', color(category.name));
    row.append(dot, node('span', '', category.title), node('span', '', count >= 6 ? `${count} confirmed` : `${count} / 6 to start`));
    $('training-counts').append(row);
  });
  const ready = state.classes.length && state.classes.every(c => (learning.counts?.[c.name] || 0) >= 6);
  $('train-head').disabled = !ready || Boolean(state.job); $('train-encoder').disabled = !ready || Boolean(state.job);
  const report = learning.report;
  $('report-panel').hidden = !report;
  $('f1-stat').textContent = report ? report.candidate.macro_f1.toFixed(2) : '-';
  if (!report) return;
  $('report-status').textContent = report.adopted ? 'Candidate adopted' : 'Baseline retained';
  $('report-summary').textContent = `${report.train_count} confirmed notes used for training · ${report.test_count} held out · baseline F1 ${report.baseline.macro_f1.toFixed(2)} → candidate F1 ${report.candidate.macro_f1.toFixed(2)}. This measures this split, not all future notes.`;
  const table = node('table'); const head = node('thead'); const heading = node('tr');
  ['Category', 'Precision', 'Recall', 'F1', 'Test notes'].forEach(text => heading.append(node('th', '', text))); head.append(heading); table.append(head);
  const body = node('tbody'); report.candidate.per_class.forEach(item => {
    const row = node('tr'); [title(item.label), item.precision, item.recall, item.f1, item.support].forEach(value => row.append(node('td', '', value))); body.append(row);
  }); table.append(body); $('report-table').replaceChildren(table);
  $('report-mistakes').replaceChildren();
  if (report.mistakes.length) $('report-mistakes').append(node('h3', '', 'Held-out mistakes'));
  report.mistakes.slice(0, 30).forEach(item => { const row = node('div', 'mistake', item.text); row.append(node('small', '', `Expected ${title(item.expected)} · predicted ${title(item.predicted)}`)); $('report-mistakes').append(row); });
}

function classEditor(category = { name: '', title: '', description: '', examples: [] }) {
  const editor = node('section', 'class-editor');
  const name = node('input'); name.value = category.name; name.required = true; name.pattern = '[a-z][a-z0-9_-]{0,39}'; name.dataset.field = 'name';
  const titleInput = node('input'); titleInput.value = category.title; titleInput.required = true; titleInput.maxLength = 60; titleInput.dataset.field = 'title';
  const formRow = node('div', 'form-row'); const nameLabel = node('label', '', 'Category ID'); nameLabel.append(name); const titleLabel = node('label', '', 'Display name'); titleLabel.append(titleInput); formRow.append(nameLabel, titleLabel);
  const descriptionLabel = node('label', '', 'Meaning'); const description = node('textarea'); description.value = category.description; description.required = true; description.maxLength = 2000; description.dataset.field = 'description'; descriptionLabel.append(description);
  const examplesLabel = node('label', '', 'Example notes · one per line'); const examples = node('textarea'); examples.value = category.examples.join('\n'); examples.rows = 4; examples.dataset.field = 'examples'; examplesLabel.append(examples);
  const remove = node('button', 'text-button', 'Remove category'); remove.type = 'button'; remove.onclick = () => editor.remove();
  editor.append(formRow, descriptionLabel, examplesLabel, remove); $('class-editors').append(editor);
}

$('import').onclick = $('empty-import').onclick = openImport;
document.querySelectorAll('[data-close]').forEach(button => button.onclick = () => $(button.dataset.close).close());
document.querySelectorAll('[data-view]').forEach(button => button.onclick = () => { if (button.dataset.view === 'notes') state.filter = 'all'; setView(button.dataset.view); });
document.querySelectorAll('[data-input]').forEach(button => button.onclick = () => setInput(button.dataset.input));
$('review-filter').onclick = () => { state.filter = 'review'; state.page = 0; setView('notes'); };
$('search').oninput = () => { clearTimeout(state.searchTimer); state.searchTimer = setTimeout(() => { state.page = 0; render(); }, 120); };
$('highlights').onchange = render;
$('board-view').onclick = () => setLayout('board'); $('table-view').onclick = () => setLayout('table');
function setLayout(layout) { state.layout = layout; ['board', 'table'].forEach(name => { $(name + '-view').classList.toggle('active', layout === name); $(name + '-view').setAttribute('aria-pressed', layout === name); }); render(); }
$('previous-page').onclick = () => { state.page--; render(); }; $('next-page').onclick = () => { state.page++; render(); };
$('move-selected').onclick = () => run(() => move([...state.selected], $('bulk-category').value));
$('confirm-selected').onclick = () => run(async () => {
  const ids = new Set(state.selected);
  for (const category of state.classes) {
    const selected = state.notes.filter(n => ids.has(n.id) && n.label === category.name).map(n => n.id);
    if (selected.length) await request('/api/move', { ids: selected, label: category.name });
  }
  state.selected.clear(); await refresh(); toast('Selected predictions confirmed and saved for learning.');
});
$('group-selected').onclick = () => run(async () => {
  await request('/api/group', { ids: [...state.selected], label: $('bulk-category').value });
  state.selected.clear(); await refresh(); toast('Fragments grouped as one note. Undo move restores them.');
});
$('clear-selection').onclick = () => { state.selected.clear(); render(); };
$('undo').onclick = () => run(async () => { await request('/api/undo', {}); await refresh(); toast('Move undone; correction restored.'); });
$('pick-files').onclick = () => $('file-picker').click(); $('pick-directory').onclick = () => $('directory-picker').click();
$('file-picker').onchange = event => run(() => importFiles(event.target.files)); $('directory-picker').onchange = event => run(() => importFiles(event.target.files));
$('dropzone').ondragover = event => { event.preventDefault(); $('dropzone').classList.add('dragover'); };
$('dropzone').ondragleave = () => $('dropzone').classList.remove('dragover');
$('dropzone').ondrop = event => { event.preventDefault(); $('dropzone').classList.remove('dragover'); run(() => importFiles(event.dataTransfer.files)); };
$('browse').onclick = () => run(browse);
$('import-form').onsubmit = event => {
  event.preventDefault();
  run(async () => {
    const body = { mode: $('split-mode').value, anchor: $('anchor').value,
      context: $('import-context').value, neighbor_words: Number($('import-neighbors').value) };
    if (state.input === 'paste') { if (!$('paste-text').value.trim()) throw new Error('Add some notes first'); body.text = $('paste-text').value; }
    if (state.input === 'files') { if (!state.documents.length) throw new Error('Choose files first'); body.documents = state.documents; }
    if (state.input === 'local') { body.kind = 'folder'; body.path = $('local-path').value; }
    await startJob('/api/import', body); $('import-dialog').close(); setView('notes');
  });
};
$('reclassify').onclick = () => run(openReclassify);
$('engine').onchange = () => { const local = $('engine').value !== 'small'; $('model-field').hidden = !local; $('local-model-help').hidden = !local; };
$('reclassify-form').onsubmit = event => {
  event.preventDefault(); run(async () => {
    const ids = state.selected.size ? [...state.selected] : visibleNotes().map(n => n.id);
    if (!ids.length) throw new Error('No notes selected');
    await startJob('/api/reclassify', { ids, engine: $('engine').value, model: $('local-model').value,
      include_confirmed: $('include-confirmed').checked, context: $('reclassify-context').value,
      neighbor_words: Number($('reclassify-neighbors').value) });
    $('reclassify-dialog').close();
  });
};
$('cancel-job').onclick = () => run(() => request('/api/cancel', { job: state.job }));
$('preview-kind').onchange = renderPreview;
$('save-files').onclick = () => run(async () => { await request('/api/output-settings', { targets: outputTargets() }); await request('/api/save-outputs', { overwrite: $('overwrite').checked }); await loadOutputs(); toast('Category files saved.'); });
$('download-zip').onclick = () => run(() => download('/api/download', 'organized-notes.zip'));
$('download-labels').onclick = () => run(() => download('/api/training-data', 'confirmed-notes.jsonl'));
$('train-head').onclick = () => run(() => startJob('/api/train', { kind: 'head' }));
$('train-encoder').onclick = () => run(() => startJob('/api/train', { kind: 'encoder' }));
$('manage').onclick = () => { showError(''); $('class-editors').replaceChildren(); state.classes.forEach(classEditor); $('classes-dialog').showModal(); };
$('add-class').onclick = () => classEditor();
$('classes-form').onsubmit = event => {
  event.preventDefault(); run(async () => {
    const classes = [...document.querySelectorAll('.class-editor')].map(editor => Object.fromEntries(['name', 'title', 'description', 'examples'].map(field => {
      const value = editor.querySelector(`[data-field=${field}]`).value.trim();
      return [field, field === 'examples' ? value.split('\n').map(v => v.trim()).filter(Boolean) : value];
    })));
    await request('/api/classes', { classes }); await refresh(); $('classes-dialog').close(); toast('Categories saved. Reclassify notes to apply the new definitions.');
  });
};
const today = new Date(); $('anchor').value = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}`;
run(async () => { const config = await request('/api/config'); state.token = config.token; await refresh(); });
