package dev.notechat

import java.text.Normalizer
import java.util.Locale

class WordPiece(private val vocab: Map<String, Int>) {
    fun tokens(text: String): LongArray {
        val plain = Normalizer.normalize(text.lowercase(Locale.ROOT), Normalizer.Form.NFD)
        val cleaned = StringBuilder()
        plain.codePoints().forEach { cp ->
            val type = Character.getType(cp)
            when {
                type == Character.NON_SPACING_MARK.toInt() || cp == 0 || cp == 0xfffd -> Unit
                Character.isWhitespace(cp) || type == Character.SPACE_SEPARATOR.toInt() -> cleaned.append(' ')
                Character.isISOControl(cp) -> Unit
                isCjk(cp) || type in setOf(Character.CONNECTOR_PUNCTUATION.toInt(), Character.DASH_PUNCTUATION.toInt(),
                    Character.START_PUNCTUATION.toInt(), Character.END_PUNCTUATION.toInt(), Character.OTHER_PUNCTUATION.toInt(),
                    Character.INITIAL_QUOTE_PUNCTUATION.toInt(), Character.FINAL_QUOTE_PUNCTUATION.toInt()) || cp in 33..47 || cp in 58..64 || cp in 91..96 || cp in 123..126 -> cleaned.append(' ').appendCodePoint(cp).append(' ')
                else -> cleaned.appendCodePoint(cp)
            }
        }
        val ids = mutableListOf<Long>()
        for (word in cleaned.toString().trim().split(Regex("\\s+")).filter { it.isNotBlank() }) {
            if (word.codePointCount(0, word.length) > 100) { ids.add(100); continue }
            var start = 0
            val pieces = mutableListOf<Long>()
            var failed = false
            while (start < word.length) {
                var end = word.length
                var match: Int? = null
                while (end > start) {
                    match = vocab[(if (start == 0) "" else "##") + word.substring(start, end)]
                    if (match != null) break
                    end--
                }
                if (match == null) { failed = true; break }
                pieces.add(match.toLong())
                start = end
            }
            ids.addAll(if (failed) listOf(100L) else pieces)
        }
        return ids.toLongArray()
    }
    private fun isCjk(cp: Int) = cp in 0x4e00..0x9fff || cp in 0x3400..0x4dbf || cp in 0x20000..0x2a6df ||
        cp in 0x2a700..0x2b73f || cp in 0x2b740..0x2b81f || cp in 0x2b820..0x2ceaf || cp in 0xf900..0xfaff || cp in 0x2f800..0x2fa1f
}
