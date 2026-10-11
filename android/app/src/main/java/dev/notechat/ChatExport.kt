package dev.notechat

import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object ChatExport {
    fun render(title: String, messages: List<Message>, categories: List<Category>, markdown: Boolean): String {
        val timestamp = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.ROOT)
        val titles = categories.associate { it.id to it.title }
        val output = StringBuilder(if (markdown) "# $title\n\n" else "")
        for (m in messages.filter { it.deleted == null }.sortedBy { it.created }) {
            val tag = titles[m.category] ?: "Unsorted"
            val time = timestamp.format(Date(m.created))
            if (!markdown) {
                val date=SimpleDateFormat("dd/MM/yyyy, HH:mm",Locale.ROOT).format(Date(m.created))
                output.append("$date - ${m.sender?.replace('\n',' ') ?: "Me"}: ")
                output.append(m.text)
                m.attachment?.let { a -> if(m.text.isNotEmpty())output.append('\n'); output.append("${a.name} (file attached)") }
                output.append('\n')
                continue
            }
            output.append(if (m.category == "todos") "- [${if (m.done) "x" else " "}] " else "- ")
            output.append(m.text.ifBlank { m.attachment?.name ?: "" })
            m.attachment?.let { a ->
                val name = mediaName(a)
                output.append("\n  [${a.name.replace("]", "\\]")}]($name)")
            }
            output.append("\n  $time · $tag\n\n")
        }
        return output.toString()
    }
    fun mediaName(a: Attachment): String = "media/${a.file}-${a.name.replace(Regex("[^A-Za-z0-9._-]"), "_").take(100)}"
}
