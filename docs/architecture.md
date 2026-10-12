# Architecture

## Android

`MainActivity` coordinates screens and background operations. `MessageRows` applies RecyclerView diffs and selection changes without replacing the list. `ChatViewport` owns the message-list controls and reply highlight. `ReplyPreview` renders quoted messages and `TodoRow` handles checklist interaction. Message operations live in `ChatEngine`; serialization lives in `StateJson`, and `Vault` handles encrypted persistence.

Category chats, private chats, Inbox, Unsorted, and Starred filter one message collection. A two-tab home separates category views from private chats. Inbox, its preview, and Inbox search exclude private messages by default; an explicit Display preference includes them. `NoteBoard` uses a two-column GridLayoutManager with full-width Inbox/section rows and nested horizontal preview lists. Notes pages reuse media and classification actions in a document layout with inline editing. Private chats are excluded from classification and Unsorted until explicitly categorized or promoted. Approvals determine Unsorted membership. Category changes, edits, deletion, and recovery therefore apply to every view without duplicating messages.

The encrypted snapshot holds messages, labels, preferences, training weights, per-notebook drafts and editor cursors, per-screen scroll anchors, the last tab, and walkthrough progress. `GuideCoach` remains visible while navigating actual feature screens and highlights targets after layout. Pausing returns to the prior screen; guided navigation does not insert demo records or change settings automatically. Portable backups omit transient UI state. Attachments are encrypted separately. Android Keystore wraps the data key. App lock requires strong biometrics or device credentials and clears decrypted state on backgrounding. Widget contents are hidden while locked by default. An explicit home-screen visibility setting publishes only active tasks to a separate device-encrypted projection; widget actions open the authenticated app.

Completed to-dos and scheduled starred archives use timestamp-based 24-hour visibility rules. Recently deleted retains records and media for less than 90 days. Expired records are purged when the workspace opens.

## Classification

The built-in pipeline combines intent cues, hashed wording features, and memory of approved examples. Optional quantized MiniLM adds sentence embeddings; encoder weights remain frozen on Android. Images without reviewed OCR text and recordings without reviewed transcripts are manually tagged and excluded from text training.

Thumbs-up and manual category assignment supply labels. Rejection changes the candidate tag without training. Editing, deleting, splitting, or merging approved text revokes the affected training example. Manual retagging replaces its label with a new confirmation. Batch reclassification clears rejected candidates and classifies active, unapproved trainable messages from scratch, including reviewed transcripts and OCR. Classification can be disabled entirely.

Six distinct approvals per category enable head training. Duplicate text is removed and approximately 20% of examples are held out using stable split assignments. Evaluation excludes held-out text from correction memory, seeds, and fitting. Candidate weights are adopted only when held-out macro F1 meets the starter baseline. This score measures the available labelled sample, not guaranteed accuracy on future notes.

Optional Vosk transcription runs on demand, decoding audio in bounded chunks on a worker thread. Reviewed transcripts remain attached to their recordings and become searchable text.

## Markdown files

A user-selected Android document tree contains a dedicated `Note Chat` subfolder. Each category and private chat has one Markdown file; unassigned category/inbox messages use `Unsorted.md`. `workspace.json` retains metadata, labels, personal weights, and recovery state. `_attachments` retains media. Selecting the same folder after reinstall restores the workspace; prior Markdown-only folders recover text without approval history. Stable message markers and Obsidian block IDs preserve identity independently of filenames or timestamps. Unmarked paragraphs become new messages. Text edits revoke approvals; moving a block into a category confirms its label; task checkboxes update completion. Missing blocks in an existing managed file soft-delete their records. Removing a whole file recreates it from the workspace.

The encrypted workspace retains a baseline for three-way reconciliation. Concurrent edits preserve an external conflict copy and retain the app version of the conflicting record. IDs and pending file versions are persisted before writing, allowing interrupted writes to retry without duplicating additions. Files with damaged markers or duplicate IDs stop synchronization. Folder access is limited to the document tree granted by the user. Private encrypted saves remain prompt. Markdown work is queued after 30 seconds without interaction, when the app leaves the foreground, or through Sync now. Idle checks also reconcile external edits. Failed writes remain pending for retry; no folder work runs on each tap or keystroke. External files and attachments are readable copies outside app-lock protection.

`[[File]]`, `[[File|label]]`, and standard Markdown links can navigate category files or open other Markdown files through Android. A category-file block reference can jump to its original message.

A shareable model report contains feature layout, aggregate approval counts, the active head type, and evaluation metrics. It excludes message text, attachments, category metadata, training assignments, embeddings, and trained weights. It is an inspection report, not a classifier backup.

## Backups, MODs, and measurements

Portable ZIP archives derive an AES-256 key with PBKDF2-HMAC-SHA256 (600,000 iterations and a random salt). Each compressed workspace/media entry uses its own AES-GCM nonce and authenticated entry name. Restore validates archive paths, IDs, bounds, and weight dimensions before merging. Duplicate message IDs keep the existing record. Drive is a destination in Android's picker/share flow; it does not provide automatic device synchronization.

Model-only ZIPs exclude messages, correction memory, media, drafts, and training split keys. They include category definitions and the personal head and require matching feature signatures. They are password-protected because learned weights can still contain information about training data.

Optional translation and Latin-script OCR use Google ML Kit. Language models and the Play services OCR/language-identification modules download on demand. Google processes input locally and receives SDK utilization/performance metrics under its own terms. Vosk remains the optional local English speech engine. Reminders use inexact AlarmManager alarms with opaque IDs and timestamps; notifications do not expose note text. English time expressions can schedule a to-do, or a date/time picker can set any message reminder.

A ten-note quiz freezes predictions before users label examples. Answers affect learning only when explicitly saved as confirmed notes. Session diagnostics report bounded classifier-call median/p95 latency and whole-app PSS; these exclude UI latency and are not universal performance claims.

## Desktop and v1

The desktop organizer uses SQLite for notes and corrections, a loopback HTTP server, and a browser board/table interface. It supports editable categories, neighboring context, category-file export, a trainable intent head, and optional encoder fine-tuning. Android and desktop workspaces are independent.

The v1 archive contains TF-IDF features, a calibrated LinearSVC, question/action rules, and deadline parsing. Training datasets and superseded model bundles are omitted from the public tree.

## Verification

Domain tests cover approval boundaries, import preservation, split/merge, archive deadlines, and state compatibility. Robolectric tests exercise native layouts, selection, reply navigation, scroll restoration, and playback controls. CI runs unit tests, Android lint, desktop tests, and JavaScript syntax checks. Fingerprint, recording, playback, sharing, and widget behavior also require physical-device testing.
