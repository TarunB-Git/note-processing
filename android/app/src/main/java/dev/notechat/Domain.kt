package dev.notechat

import java.util.Locale
import java.util.UUID

const val ARCHIVE_DELAY_MS = 24L * 60 * 60 * 1000

const val RETENTION_MS = 90L * 24 * 60 * 60 * 1000

data class Category(val id: String, val title: String, val description: String,
                    val examples: List<String> = emptyList(), val pinned: Boolean = false, val icon: String? = null, val photo: Attachment? = null)
data class Chat(val id: String, val title: String, val icon: String? = null, val photo: Attachment? = null)
data class QuizItem(val text: String, val prediction: String?, val answer: String? = null, val nanos: Long = 0)
data class Quiz(val items: List<QuizItem>, val memoryKb: Int = 0)
data class Attachment(val file: String, val name: String, val mime: String, val bytes: Long, val duration: Long = 0, val waveform: List<Int> = emptyList())
data class Message(
    val id: String = UUID.randomUUID().toString(), val text: String = "", val created: Long = System.currentTimeMillis(),
    val category: String? = null, val approved: Boolean = false, val rejected: Set<String> = emptySet(),
    val attachment: Attachment? = null, val replyTo: String? = null, val starred: Boolean = false,
    val done: Boolean = false, val edited: Boolean = false, val deleted: Long? = null,
    val reason: String = "", val manual: Boolean = false, val sender: String? = null,
    val completedAt: Long? = null, val starArchiveAt: Long? = null, val transcribed: Boolean = false,
    val pinnedUntil: Long? = null, val chatId: String? = null, val remindAt: Long? = null,
    val translation: String? = null, val extracted: Boolean = false
)
data class TrainingReport(val confirmed: Int, val heldOut: Int, val baseline: Double, val candidate: Double,
                          val adopted: Boolean, val timestamp: Long = System.currentTimeMillis())
data class PersonalHead(val signature: String, val weights: List<List<Float>>, val bias: List<Float>)
data class Preferences(val preserveImportDates: Boolean = false, val appearance: String = "system", val widgetWhenLocked: Boolean = false, val classification: Boolean = true, val frontend: String = "chat",
                       val autoTranslate: Boolean = false, val targetLanguage: String = "en", val ocr: Boolean = false, val reminders: Boolean = false, val includePrivateChatsInInbox: Boolean = false)
data class MarkdownState(val tree: String? = null, val baseline: Map<String, String> = emptyMap(), val synced: Long = 0, val pendingExternal: Map<String,String> = emptyMap(), val pendingRendered: Map<String,String> = emptyMap())

fun Message.completedArchived(now: Long = System.currentTimeMillis()) = done && completedAt != null && now - completedAt >= ARCHIVE_DELAY_MS
fun Message.starArchived(now: Long = System.currentTimeMillis()) = starred && starArchiveAt != null && now - starArchiveAt >= ARCHIVE_DELAY_MS
fun Message.visible(now: Long = System.currentTimeMillis()) = deleted == null && !completedArchived(now) && !starArchived(now)
fun Message.inInbox(preferences: Preferences) = chatId == null || preferences.includePrivateChatsInInbox
fun Message.trainable() = chatId == null && (attachment == null || (text.isNotBlank() && (attachment.mime.startsWith("audio/") && transcribed || attachment.mime.startsWith("image/") && extracted)))

data class Navigation(val route: String = "home", val anchor: String? = null, val index: Int = 0, val offset: Int = 0, val scroll: Int = 0)

data class ChatState(val categories: List<Category>, val messages: List<Message> = emptyList(), val report: TrainingReport? = null,
                     val model: PersonalHead? = null, val splits: Map<String, String> = emptyMap(),
                     val draft: String = "", val replyDraft: String? = null, val preferences: Preferences = Preferences(), val navigation: Navigation = Navigation(), val markdown: MarkdownState = MarkdownState(), val chats: List<Chat> = emptyList(), val importedArchives: Set<String> = emptySet(), val quiz: Quiz? = null, val editDraftId: String? = null, val preEditDraft: String = "", val drafts: Map<String,ComposerDraft> = emptyMap(), val locations: Map<String,Navigation> = emptyMap(), val guide: GuideProgress = GuideProgress(), val homeTab: String = "home")
data class Suggestion(val category: String?, val reason: String)

fun normalized(text: String) = text.trim().replace(Regex("\\s+"), " ").lowercase(Locale.ROOT)

class ChatEngine(var state: ChatState) {
    fun pin(id: String, until: Long?, now: Long = System.currentTimeMillis()) = update(id) {
        require(it.visible(now)) { "Only active messages can be pinned" }
        require(until == null || until > now)
        require(until == null || it.pinnedUntil?.let { expiry -> expiry > now } == true ||
            state.messages.count { message -> message.visible(now) && (message.pinnedUntil ?: 0) > now } < 3) { "Unpin a message first; up to three messages can be pinned." }
        it.copy(pinnedUntil = until)
    }
    fun addChat(chat: Chat) {
        require(state.chats.size<256 && chat.title.isNotBlank() && chat.title.length<=60 && state.chats.none { it.id==chat.id || it.title.equals(chat.title,true) })
        state=state.copy(chats=state.chats+chat)
    }
    fun promoteChat(id: String, description: String) {
        val chat=state.chats.first { it.id==id }
        addCategory(Category(chat.id,chat.title,description,icon=chat.icon,photo=chat.photo))
        state=state.copy(chats=state.chats.filterNot { it.id==id },messages=state.messages.map {
            if(it.chatId==id)it.copy(chatId=null,category=id,approved=true,manual=true,reason="Chat added to categories by you.") else it
        })
    }
    fun forward(ids: Set<String>, destination: String): List<Message> {
        val chat=destination.removePrefix("chat:").takeIf { destination.startsWith("chat:") }
        val category=destination.takeUnless { it=="all" || chat!=null }
        require(chat==null || state.chats.any { it.id==chat })
        require(category==null || state.categories.any { it.id==category })
        val copies=state.messages.filter { it.id in ids && it.visible() }.map {
            it.copy(id=UUID.randomUUID().toString(),created=System.currentTimeMillis(),category=category,chatId=chat,approved=category!=null,manual=category!=null,
                replyTo=null,rejected=emptySet(),done=false,completedAt=null,starred=false,starArchiveAt=null,pinnedUntil=null,remindAt=null,reason="Forwarded by you.")
        }
        require(copies.size==ids.size)
        state=state.copy(messages=state.messages+copies)
        return copies
    }
    fun active() = state.messages.filter { it.deleted == null }
    fun confirmed() = active().filter { it.approved && it.trainable() && it.category != null }
        .asReversed().distinctBy { normalized(it.text) }
    fun update(id: String, action: (Message) -> Message) {
        require(state.messages.any { it.id == id }) { "Message no longer exists" }
        val old = state.messages.first { it.id == id }
        val changed = action(old).let { if(it.category != "todos")it.copy(done=false,completedAt=null) else it }
        val revoked = old.approved && (!changed.approved || old.text != changed.text || old.category != changed.category || changed.deleted != null)
        state = state.copy(messages = state.messages.map { if (it.id == id) changed else it },
            model = if (revoked) null else state.model, report = if (revoked) null else state.report)
    }
    fun approve(id: String) = update(id) {
        require(it.deleted == null && it.chatId == null && state.categories.any { c -> c.id == it.category }) { "Choose a category first" }
        it.copy(approved = true)
    }
    fun reject(id: String, predict: (Message) -> Suggestion) = update(id) {
        require(it.deleted == null)
        val rejected = it.rejected + listOfNotNull(it.category)
        val pending = it.copy(approved = false, rejected = rejected, category = null, manual = false)
        val next = predict(pending)
        require(next.category !in rejected) { "A rejected category cannot be suggested again" }
        pending.copy(category = next.category, reason = next.reason)
    }
    fun tag(id: String, category: String?) = update(id) {
        require(category == null || state.categories.any { c -> c.id == category })
        it.copy(category = category, chatId = null, approved = category != null, rejected = emptySet(), manual = true, reason = if(category==null)"Category removed." else "Category confirmed by you.")
    }
    fun edit(id: String, text: String) = update(id) {
        require(text.isNotBlank() && text.length <= 20000)
        it.copy(text = text, category = if(it.trainable())null else it.category, approved = false, rejected = emptySet(), edited = true, manual = if(it.trainable())false else it.manual)
    }
    fun complete(id: String, done: Boolean, now: Long = System.currentTimeMillis()) = update(id) {
        require(it.deleted == null && it.category == "todos")
        it.copy(done = done, completedAt = if (done) now else null)
    }
    fun archiveStars(ids: Set<String>, now: Long = System.currentTimeMillis()) {
        ids.forEach { id -> update(id) { require(it.starred && it.deleted == null); it.copy(starArchiveAt = now) } }
    }
    fun deleteRange(from: Long, through: Long, now: Long = System.currentTimeMillis()) {
        require(from <= through) { "End must follow start" }
        delete(active().filter { it.created in from..through }.map { it.id }.toSet(), now)
    }
    fun split(id: String, parts: List<String>): List<Message> {
        val original = active().first { it.id == id }
        require(original.attachment == null) { "Split text messages only" }
        require(parts.isNotEmpty() && parts.all { it.isNotBlank() && it.length <= 20000 })
        val replacements = parts.mapIndexed { index, text -> original.copy(
            id = if (index == 0) id else UUID.randomUUID().toString(), text = text,
            category = null, approved = false, rejected = emptySet(), manual = false,
            edited = true, reason = "", done = false, completedAt = null
        ) }
        state = state.copy(messages = state.messages.flatMap { if (it.id == id) replacements else listOf(it) }, model = null, report = null)
        return replacements
    }
    fun merge(ids: Set<String>): Message {
        val records = active().filter { it.id in ids }
        require(records.size >= 2 && records.size == ids.size) { "Select at least two active messages" }
        require(records.map { it.chatId }.distinct().size==1) { "Merge messages from the same chat" }
        require(records.all { it.attachment == null }) { "Merge text messages only; attachments must stay separate" }
        val text = records.joinToString("\n\n") { it.text }
        require(text.length <= 20000) { "Merged text exceeds 20,000 characters" }
        val first = records.first()
        val merged = first.copy(text = text, category = null, approved = false, rejected = emptySet(),
            manual = false, edited = true, reason = "", done = false, completedAt = null,
            starred = records.any { it.starred }, starArchiveAt = null)
        state = state.copy(messages = state.messages.mapNotNull {
            when { it.id == first.id -> merged; it.id in ids -> null
                it.replyTo in ids -> it.copy(replyTo = first.id); else -> it }
        }, model = null, report = null)
        return merged
    }
    fun delete(ids: Set<String>, now: Long = System.currentTimeMillis()) {
        state = state.copy(messages = state.messages.map { if (it.id in ids && it.deleted == null) it.copy(deleted = now) else it }, model = null, report = null)
    }
    fun restore(id: String, now: Long = System.currentTimeMillis()) = update(id) {
        require(it.deleted != null && now - it.deleted < RETENTION_MS) { "The recovery period has expired" }
        it.copy(deleted = null)
    }
    fun expire(now: Long = System.currentTimeMillis()): List<Message> {
        val expired = state.messages.filter { it.deleted != null && now - it.deleted >= RETENTION_MS }
        val ids = expired.map { it.id }.toSet()
        state = state.copy(messages = state.messages.filterNot { it.id in ids },
            splits = state.splits.filterKeys { key -> expired.none { normalized(it.text) == key } })
        return expired
    }
    fun reclassifyPending(now: Long = System.currentTimeMillis(), predict: (Message) -> Suggestion): Int {
        if(!state.preferences.classification)return 0
        var count = 0
        val messages = state.messages.map { original ->
            if (!original.visible(now) || original.approved || !original.trainable()) original
            else {
                val pending = original.copy(category = null, rejected = emptySet(), manual = false)
                val suggestion = predict(pending)
                require(suggestion.category == null || state.categories.any { it.id == suggestion.category })
                count++
                pending.copy(category = suggestion.category, reason = suggestion.reason,
                    done = original.done && suggestion.category == "todos",
                    completedAt = original.completedAt.takeIf { suggestion.category == "todos" })
            }
        }
        state = state.copy(messages = messages)
        return count
    }
    fun removeCategory(id: String, replacement: String?) {
        require(state.categories.any { it.id == id })
        require(replacement == null || state.categories.any { it.id == replacement && it.id != id })
        state = state.copy(categories = state.categories.filterNot { it.id == id }, report = null, model = null, quiz = null,
            messages = state.messages.map { if (it.category == id)
                it.copy(category = replacement, approved = replacement!=null && it.chatId==null, rejected = emptySet(), manual = true,reason=if(replacement==null)"Category removed." else "Category confirmed by you.",
                    done = if(replacement=="todos")it.done else false, completedAt = if(replacement=="todos")it.completedAt else null)
                else it.copy(rejected = it.rejected - id) })
    }
    fun addCategory(category: Category) {
        require(category.title.isNotBlank() && category.title.length <= 60 && category.description.isNotBlank() && category.description.length<=2000 && category.examples.all { it.length<=2000 })
        require(state.categories.size < 32 && state.categories.none { it.id == category.id || it.title.equals(category.title, true) })
        state = state.copy(categories = state.categories + category, report = null, model = null, quiz = null)
    }
}
