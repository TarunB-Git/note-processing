# Architecture

## Android

`MainActivity` coordinates screens and background operations. `ChatViewport` owns the message-list controls and reply highlight. `ReplyPreview` renders quoted messages and `TodoRow` handles checklist interaction. Message operations live in `ChatEngine`; serialization lives in `StateJson`, and `Vault` handles encrypted persistence.

Category chats, Inbox, Unsorted, and Starred filter one message collection. Approvals determine Unsorted membership. Category changes, edits, deletion, and recovery therefore apply to every view without duplicating messages.

The encrypted snapshot holds messages, labels, preferences, training weights, drafts, and the last screen and message anchor. Attachments are encrypted separately. Android Keystore wraps the data key. App lock requires strong biometrics or device credentials and clears decrypted state on backgrounding. Widget contents are hidden while locked.

Completed to-dos and scheduled starred archives use timestamp-based 24-hour visibility rules. Recently deleted retains records and media for less than 90 days. Expired records are purged when the workspace opens.

## Classification

The built-in pipeline combines intent cues, hashed wording features, and memory of approved examples. Optional quantized MiniLM adds sentence embeddings; encoder weights remain frozen on Android. Images and recordings without reviewed transcripts are manually tagged and excluded from text training.

Approvals supply labels. Rejection changes the candidate tag without training. Editing, retagging, deleting, splitting, or merging approved text revokes the affected training example. Batch reclassification changes only active, unapproved text and reviewed voice transcripts.

Six distinct approvals per category enable head training. Duplicate text is removed and approximately 20% of examples are held out using stable split assignments. Evaluation excludes held-out text from correction memory, seeds, and fitting. Candidate weights are adopted only when held-out macro F1 meets the starter baseline. This score measures the available labelled sample, not guaranteed accuracy on future notes.

Optional Vosk transcription runs on demand, decoding audio in bounded chunks on a worker thread. Reviewed transcripts remain attached to their recordings and become searchable text.

## Desktop and v1

The desktop organizer uses SQLite for notes and corrections, a loopback HTTP server, and a browser board/table interface. It supports editable categories, neighboring context, category-file export, a trainable intent head, and optional encoder fine-tuning. Android and desktop workspaces are independent.

The v1 archive contains TF-IDF features, a calibrated LinearSVC, question/action rules, and deadline parsing. Training datasets and superseded model bundles are omitted from the public tree.

## Verification

Domain tests cover approval boundaries, import preservation, split/merge, archive deadlines, and state compatibility. Robolectric tests exercise native layouts, selection, reply navigation, scroll restoration, and playback controls. CI runs unit tests, Android lint, desktop tests, and JavaScript syntax checks. Fingerprint, recording, playback, sharing, and widget behavior also require physical-device testing.
