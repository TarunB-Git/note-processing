package dev.notechat

object GuidePages {
    data class Step(val title: String, val text: String, val route: String, val target: String)
    val steps = listOf(
        Step("Capture a note", "Type in the highlighted field, then send. The Inbox keeps your original order. Your existing draft stays here until you send or edit it.", "all", "composer"),
        Step("Confirm a suggestion", "Use thumbs-up to confirm the category. Thumbs-down tries another label without teaching it. A manual category choice also confirms your label.", "all", "feedback"),
        Step("Choose a category", "Tap the category under a note to change it. Long-press a note for multiple selection, then tag or approve the selected notes together.", "all", "tag"),
        Step("Forward a note", "Tap the forward arrow to the left of a message. Choose a category, private chat, or WhatsApp. The original remains where it was.", "all", "forward"),
        Step("Private chats", "Chats is the right-hand tab. Tap the plus to create a private chat. These messages are ignored by the classifier; a chat's menu can make it a category.", "chats", "new-chat"),
        Step("Your categories", "Add a category with a short description and examples. Its menu also changes the icon, pins it, exports it, or adds a home-screen shortcut.", "categories", "add-category"),
        Step("Keep Markdown files", "Choose Permanent Markdown folder. Files survive uninstall until you remove them. Select the same folder after reinstall to restore. Saves wait for 30 seconds idle or backgrounding.", "settings:data", "markdown"),
        Step("Optional tools", "Voice transcription, OCR, translation, and reminders are optional. Downloads and consent happen only when you choose them. Google ML Kit processes content locally and sends SDK usage metrics.", "mods", "speech"),
        Step("Check the learner", "The small learner is already on your phone. Test with 10 fresh notes freezes predictions before you label them. Train & compare needs six distinct approvals per category; MiniLM is optional.", "model", "test-model"),
        Step("Choose your layout", "Layout switches between chat and notes. Notes has one Inbox page, two category boards per row, and private-chat boards below. Swipe a board preview to browse its recent notes.", "settings:display", "layout"),
        Step("Complete a task", "Tap a task or its checkbox to complete it. Completed tasks move to the Settings archive after one day. Long-press still allows tagging and forwarding.", "todos", "task"),
        Step("Make a backup", "Create an encrypted ZIP and keep its password. Choose device storage or Drive. Restore skips existing message IDs. The Markdown folder is readable; encrypted ZIPs protect backups you share.", "settings:data", "backup")
    )
}
