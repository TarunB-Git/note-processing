package dev.notechat

import java.text.SimpleDateFormat
import java.util.Locale

object ChatImport {
    data class Entry(val text: String, val created: Long, val sender: String? = null)
    data class Result(val entries: List<Entry>, val transcript: Boolean)
    private val header = Regex("^\\[?(\\d{1,4}[./-]\\d{1,2}[./-]\\d{1,4}),?\\s+(\\d{1,2}:\\d{2}(?::\\d{2})?(?:\\s*[AaPp][Mm])?)\\]?\\s*(?:-\\s*)?(.+)$")

    fun parse(source: String, dayFirst: Boolean = true, now: Long = System.currentTimeMillis()): Result {
        val text = source.removePrefix("\uFEFF").replace("\r\n", "\n").replace('\r', '\n')
        val rows = mutableListOf<Entry>()
        var current: Entry? = null
        val leading=StringBuilder()
        var transcriptFence:Char?=null
        for (raw in text.lines()) {
            val line = raw.replace(Regex("[\u200e\u200f\u202a-\u202e\u2066-\u2069]"), "").replace('\u202f', ' ')
            val marker=Regex("^\\s*(```+|~~~+)").find(line)?.groupValues?.get(1)?.first()
            val match = if(transcriptFence==null)header.matchEntire(line) else null
            if(marker!=null)transcriptFence=if(transcriptFence==marker)null else marker
            val date = match?.let { date(it.groupValues[1], it.groupValues[2], dayFirst) }
            if (match != null && date != null) {
                current?.let { rows.add(it) }
                val content = match.groupValues[3]
                val colon = content.indexOf(": ")
                current = Entry(if (colon >= 0) content.substring(colon + 2) else content, date,
                    if (colon >= 0) content.substring(0, colon) else null)
            } else if (current != null) current = current.copy(text = current.text + "\n" + raw)
            else { if(leading.isNotEmpty())leading.append('\n'); leading.append(raw) }
        }
        current?.let { rows.add(it) }
        if (rows.isNotEmpty()) return Result((if(leading.isNotBlank())split(Entry(leading.toString().trim(),now)) else emptyList())+rows.flatMap { split(it) }, true)
        // Blank paragraphs separate notes; fenced code and Markdown lists remain intact.
        val blocks = mutableListOf<String>()
        val block = StringBuilder()
        var fence: String? = null
        for (line in text.lines()) {
            val marker = Regex("^\\s*(```+|~~~+)").find(line)?.groupValues?.get(1)
            if (marker != null) fence = if (fence?.first() == marker.first()) null else marker
            if (line.isBlank() && fence == null) {
                if (block.isNotBlank()) { blocks.add(block.toString().trim()); block.clear() }
            } else { if (block.isNotEmpty()) block.append('\n'); block.append(line) }
        }
        if (block.isNotBlank()) blocks.add(block.toString().trim())
        return Result(blocks.flatMapIndexed { index, body -> split(Entry(body, now + index)) }, false)
    }
    fun messages(entries:List<Entry>,preserveDates:Boolean=false,importedAt:Long=System.currentTimeMillis()) = entries.map {
        Message(text=it.text,created=if(preserveDates)it.created else importedAt,sender=it.sender)
    }
    private fun split(entry: Entry) = entry.text.trimEnd().chunked(20000).filter { it.isNotBlank() }.map { entry.copy(text = it) }
    private fun date(day: String, time: String, dayFirst: Boolean): Long? {
        val parts = day.split(Regex("[./-]"))
        val yearFirst = parts[0].length == 4
        val pattern = if (yearFirst) "yyyy/M/d" else if (dayFirst) "d/M/${if (parts[2].length == 2) "yy" else "yyyy"}" else "M/d/${if (parts[2].length == 2) "yy" else "yyyy"}"
        val clock = if (time.contains(Regex("[AaPp][Mm]"))) "h:mm${if (time.count { it == ':' } == 2) ":ss" else ""} a" else "H:mm${if (time.count { it == ':' } == 2) ":ss" else ""}"
        val format = SimpleDateFormat("$pattern $clock", Locale.US).apply { isLenient = false }
        val value = day.replace('.', '/').replace('-', '/') + " " + time.replace(Regex("(?i)(\\d)(am|pm)$"), "$1 $2").uppercase(Locale.US)
        val position = java.text.ParsePosition(0)
        return format.parse(value, position)?.takeIf { position.index == value.length }?.time
    }
}
