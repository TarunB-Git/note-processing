package dev.notechat

import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID
import org.json.JSONObject

/** Round-trip format shared by the folder bridge and Markdown editors. */
object MarkdownWorkspace {
    data class Entry(val id: String?, val text: String, val category: String?, val done: Boolean, val chatId: String? = null)
    data class Document(val category: String?, val entries: List<Entry>, val chatId: String? = null)
    data class Merge(val state: ChatState, val conflicts: Set<String>)
    private val start = Regex("(?m)^<!-- notechat:message ([A-Za-z0-9-]+) -->\\s*$")
    private const val end = "<!-- /notechat -->"

    fun filename(state: ChatState, category: String?, chatId: String? = null): String {
        state.markdown.baseline.forEach { (name, content) ->
            if (header(content) == (category ?: "unassigned") && chatHeader(content)==chatId) return name
        }
        if (chatId!=null)return "Chat-${state.chats.first { it.id==chatId }.title.replace(Regex("[^\\p{L}\\p{N} _-]"),"").take(45)}-${chatId.take(8)}.md"
        if (category == null) return "Unsorted.md"
        fun safe(title: String) = title.replace(Regex("[^\\p{L}\\p{N} _-]"), "").trim().take(60).ifBlank { "Category" }
        val title = safe(state.categories.first { it.id == category }.title)
        val collision = title.equals("Unsorted",true) || state.categories.count { safe(it.title).equals(title,true) } > 1
        return if(collision)"$title-${category.take(8)}.md" else "$title.md"
    }
    fun render(state: ChatState, category: String?, now: Long = System.currentTimeMillis(), chatId: String? = null): String {
        val title = if(chatId!=null)state.chats.first { it.id==chatId }.title else state.categories.firstOrNull { it.id == category }?.title ?: "Unsorted"
        val timestamp = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.ROOT)
        return buildString {
            append("---\nnotechat: 1\ncategory: ${category ?: "unassigned"}\n${if(chatId!=null)"chat: $chatId\n" else ""}aliases: [${JSONObject.quote(title)}]\n---\n# ${title.replace('\n',' ')}\n\n")
            state.messages.filter { it.visible(now) && it.category == category && it.chatId == chatId }.forEach { m ->
                require(!start.containsMatchIn(m.text) && !m.text.lines().contains(end)) { "A note contains reserved Markdown sync markers." }
                append("<!-- notechat:message ${m.id} -->\n## ${timestamp.format(Date(m.created))}\n\n")
                if (category == "todos") append("- [${if (m.done) "x" else " "}] ")
                append(m.text)
                m.attachment?.let { append("\n\n${mediaLink(it)}") }
                append("\n\n^nc-${m.id}\n$end\n\n")
            }
        }
    }
    fun mediaName(a: Attachment) = ChatExport.mediaName(a).replaceFirst("media/", "_attachments/")
    private fun mediaLink(a: Attachment) = "${if (a.mime.startsWith("image/")) "!" else ""}[${a.name.replace("]", "\\]")}](${mediaName(a)})"
    private fun chatHeader(source: String) = Regex("(?m)^chat: ([A-Za-z0-9-]+)$").find(source.substringBefore("\n---\n",source))?.groupValues?.get(1)
    fun categoryId(source: String) = header(source)
    fun chatId(source: String) = chatHeader(source)
    private fun header(source: String): String? {
        if (!source.startsWith("---\n") || !source.lineSequence().take(10).contains("notechat: 1")) return null
        return Regex("(?m)^category: ([A-Za-z0-9-]+)$").find(source.substringBefore("\n---\n",source))?.groupValues?.get(1)
    }
    fun parse(source: String, state: ChatState): Document {
        val chatId=chatHeader(source)
        require(chatId==null || state.chats.any { it.id==chatId }) { "Markdown chat no longer exists" }
        val rawCategory = header(source) ?: error("Markdown sync header is missing or damaged")
        val category = rawCategory.takeUnless { it == "unassigned" }
        require(chatId==null || category==null) { "Private Markdown chat cannot also be a category" }
        require(category == null || state.categories.any { it.id == category }) { "Markdown category no longer exists" }
        val headerEnd = source.indexOf("\n---\n", 4)
        require(headerEnd >= 0)
        val content = source.substring(headerEnd + 5).replaceFirst(Regex("^# [^\n]*\n"), "")
        val entries = mutableListOf<Entry>()
        val outside = StringBuilder()
        var offset = 0
        for (match in start.findAll(content)) {
            require(match.range.first >= offset) { "Overlapping Markdown message markers" }
            outside.append(content.substring(offset, match.range.first)).append("\n\n")
            val endIndex = content.indexOf(end, match.range.last + 1)
            require(endIndex >= 0) { "A Markdown message is missing its closing marker" }
            val id = match.groupValues[1]
            var text = content.substring(match.range.last + 1, endIndex).trim()
                .replaceFirst(Regex("^## \\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}\\s*"), "")
                .replace(Regex("(?m)^\\^nc-${Regex.escape(id)}\\s*$"), "").trim()
            state.messages.firstOrNull { it.id == id }?.attachment?.let { text = text.replace(mediaLink(it), "").trim() }
            val checkbox = Regex("^- \\[([ xX])][ \\t]?").find(text)
            if (category == "todos" && checkbox != null) text = text.substring(checkbox.range.last + 1)
            require(text.length <= 20000) { "A Markdown message exceeds 20,000 characters" }
            entries.add(Entry(id, text, category, category == "todos" && checkbox?.groupValues?.get(1)?.lowercase() == "x",chatId))
            offset = endIndex + end.length
        }
        outside.append(content.substring(offset))
        require(!outside.contains(end)) { "A Markdown message is missing its opening marker" }
        // Unmarked paragraphs appended in an editor become new messages; fenced code stays together.
        val pending = StringBuilder(); var fence: String? = null
        fun flush() { val text = pending.toString().trim(); if (text.isNotEmpty()) {
            require(text.length <= 20000)
            val check = Regex("^- \\[([ xX])][ \\t]?").find(text)
            entries.add(Entry(null,if (category == "todos" && check != null) text.substring(check.range.last+1) else text,category,category == "todos" && check?.groupValues?.get(1)?.lowercase()=="x",chatId))
        }; pending.clear() }
        outside.lines().forEach { line ->
            val delimiter = Regex("^\\s*(`{3,}|~{3,})").find(line)?.groupValues?.get(1)
            if (delimiter != null) fence = if (fence == null) delimiter else if (delimiter.first() == fence!!.first() && delimiter.length >= fence!!.length) null else fence
            if (line.isBlank() && fence == null) flush() else pending.append(line).append('\n')
        }
        flush()
        require(entries.mapNotNull { it.id }.distinct().size == entries.count { it.id != null }) { "Duplicate Markdown message IDs" }
        return Document(category,entries,chatId)
    }
    fun merge(state: ChatState, external: Map<String,String>, now: Long = System.currentTimeMillis()): Merge {
        val parsed = external.mapValues { parse(it.value,state) }
        val baseline = state.markdown.baseline.filterValues { header(it)?.let { id -> id == "unassigned" || state.categories.any { it.id == id } } == true && (chatHeader(it)==null || state.chats.any { chat -> chat.id==chatHeader(it) }) }
            .mapValues { parse(it.value,state) }
        val allEntries = parsed.values.flatMap { it.entries }
        require(allEntries.mapNotNull { it.id }.distinct().size == allEntries.count { it.id != null }) { "A message appears in more than one Markdown file" }
        val previous = baseline.values.flatMap { it.entries }.associateBy { it.id }
        val present = allEntries.mapNotNull { it.id }.toSet()
        val engine = ChatEngine(state)
        val conflicts = mutableSetOf<String>()
        parsed.forEach { (name,doc) ->
            if (external[name] == state.markdown.baseline[name]) return@forEach
            doc.entries.forEach { entry ->
                val old = entry.id?.let { id -> state.messages.firstOrNull { it.id == id } }
                if (old == null) {
                    if (entry.id != null && entry.id in previous) return@forEach
                    if (entry.text.isNotBlank()) engine.state = engine.state.copy(messages = engine.state.messages + Message(
                        id=UUID.randomUUID().toString(),text=entry.text,created=now,category=entry.category,chatId=entry.chatId,manual=entry.category!=null,approved=entry.category!=null && entry.chatId==null,
                        done=entry.done,completedAt=now.takeIf { entry.done },reason=if(entry.category==null)"Added through Markdown. Choose a category to teach it." else "Category assigned through Markdown."))
                } else {
                    val before = previous[entry.id]
                    val actual = Entry(old.id,old.text,old.category,old.done,old.chatId)
                    val changedOutside = before == null || entry != before
                    if (!changedOutside) return@forEach
                    if (!old.visible(now) || (before != null && actual != before && actual != entry)) { conflicts.add(name); return@forEach }
                    if (entry.text.isBlank() && old.attachment==null) { engine.delete(setOf(old.id),now); return@forEach }
                    val relabeled = old.text != entry.text || old.category != entry.category || old.chatId!=entry.chatId
                    engine.update(old.id) { it.copy(text=entry.text,category=entry.category,chatId=entry.chatId,approved=entry.chatId==null && (if(old.category!=entry.category)entry.category!=null else it.approved && !relabeled),
                        edited=it.edited || it.text!=entry.text,manual=it.manual || relabeled,rejected=if(relabeled)emptySet() else it.rejected,
                        done=entry.done,completedAt=if(entry.done)it.completedAt ?: now else null,
                        reason=if(relabeled)"Edited through Markdown. Tap thumbs-up to teach it." else it.reason) }
                }
            }
            baseline[name]?.entries?.mapNotNull { it.id }?.filterNot { it in present }?.forEach { id ->
                val old = state.messages.firstOrNull { it.id==id } ?: return@forEach
                if (!old.visible(now)) return@forEach
                if (Entry(old.id,old.text,old.category,old.done,old.chatId) != previous[id]) conflicts.add(name) else engine.delete(setOf(id),now)
            }
        }
        return Merge(engine.state,conflicts)
    }
}
