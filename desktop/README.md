# Desktop organizer

Local note classification, review, and category-file export using Python and a browser interface. TXT/Markdown input can be segmented by paragraph, line, or sentence, with optional neighboring context. Notes appear in a board or table and export to category files.

The pipeline combines intent cues, MiniLM embeddings, a trainable head, and confirmed-example memory. Optional encoder fine-tuning uses the same held-out adoption check. The HTTP server binds to loopback; SQLite stores notes and corrections in an ignored local workspace.

`src/note_processing/` contains the classifier, training pipeline, server, and browser UI. `tests/` covers import preservation, undo, export protection, and training boundaries. Python 3.10 or later is required. The package exposes the `note-processing` command.

[Architecture](../docs/architecture.md) · [Apache 2.0](LICENSE)
