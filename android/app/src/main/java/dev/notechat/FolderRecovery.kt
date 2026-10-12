package dev.notechat

import java.text.SimpleDateFormat
import java.util.Locale
import java.util.UUID

object FolderRecovery {
    fun reconstruct(files: Map<String,String>, templates: List<Category>, media: Map<String,Attachment> = emptyMap()): ChatState {
        val categories=templates.toMutableList(); val chats=mutableListOf<Chat>()
        fun title(source: String, fallback: String) = Regex("(?m)^# (.+)$").find(source)?.groupValues?.get(1)?.take(60) ?: fallback.removeSuffix(".md").take(60)
        files.forEach { (name,source) ->
            val id=MarkdownWorkspace.categoryId(source) ?: return@forEach
            val chat=MarkdownWorkspace.chatId(source)
            if(chat!=null) { if(chats.none { it.id==chat })chats.add(Chat(chat,title(source,name))); return@forEach }
            if(id!="unassigned" && categories.none { it.id==id })categories.add(Category(id,title(source,name),"Notes saved in ${title(source,name)}"))
        }
        val attachmentById=files.values.flatMap { source -> Regex("(?s)<!-- notechat:message ([A-Za-z0-9-]+) -->(.*?)<!-- /notechat -->").findAll(source).mapNotNull { block ->
            val filename=Regex("\\(_attachments/([^)]*)\\)").find(block.groupValues[2])?.groupValues?.get(1)
            media[filename]?.let { block.groupValues[1] to it }
        }.toList() }.toMap()
        var state=ChatState(categories,chats=chats,messages=attachmentById.map { (id,attachment) -> Message(id=id,attachment=attachment) })
        val records=files.values.flatMap { source ->
            val doc=MarkdownWorkspace.parse(source,state)
            doc.entries.map { entry ->
                val marker=entry.id?.let { "<!-- notechat:message $it -->" }
                val after=if(marker!=null)source.substringAfter(marker,"") else ""
                val date=Regex("^\\s*## (\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2})").find(after)?.groupValues?.get(1)
                val created=date?.let { runCatching { SimpleDateFormat("yyyy-MM-dd HH:mm",Locale.ROOT).apply { isLenient=false }.parse(it)?.time }.getOrNull() } ?: System.currentTimeMillis()
                Message(id=entry.id ?: UUID.randomUUID().toString(),attachment=attachmentById[entry.id],text=entry.text,category=entry.category,chatId=entry.chatId,created=created,
                    done=entry.done,completedAt=created.takeIf { entry.done },manual=entry.category!=null,reason="Recovered from Markdown. Confirm the category to teach it.")
            }
        }
        // Old category files can retain a block after it was moved. Keep every distinct text,
        // but collapse duplicate copies of the same record.
        val recovered=linkedMapOf<String,Message>()
        records.forEach { message ->
            val previous=recovered[message.id]
            if(previous!=null && previous.text!=message.text)recovered[UUID.randomUUID().toString()]=previous.copy(id=UUID.randomUUID().toString(),reason="Recovered another version from an old Markdown file.")
            recovered[message.id]=message
        }
        state=state.copy(messages=recovered.values.toList())
        return state
    }
}
