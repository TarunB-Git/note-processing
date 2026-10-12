package dev.notechat

data class ComposerDraft(val text: String = "", val replyTo: String? = null, val editId: String? = null, val previous: String = "", val cursor: Int = -1)
data class GuideProgress(val active: Boolean = false, val step: Int = 0, val returnTo: Navigation = Navigation("settings"), val returnTab: String = "home")
