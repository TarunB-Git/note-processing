package dev.notechat

object TextLinks {
    private val pattern = Regex("(?i)(?<![\\p{L}\\p{N}_@])(?:https?://[^\\s<>]+|(?:www\\.)?(?:[\\p{L}\\p{N}](?:[\\p{L}\\p{N}-]*[\\p{L}\\p{N}])?\\.)+[\\p{L}]{2,63}(?::\\d{1,5})?(?:[/#?][^\\s<>]*)?)")
    data class Link(val start: Int, val end: Int, val text: String) { val url get() = if (text.contains("://")) text else "https://$text" }
    fun find(text: String): List<Link> = pattern.findAll(text).map { match ->
        var body = match.value.trimEnd('.', ',', ';', ':', '!', '?', '"', '\'')
        while (body.endsWith(')') && body.count { it == ')' } > body.count { it == '(' }) body = body.dropLast(1)
        while (body.endsWith(']') && body.count { it == ']' } > body.count { it == '[' }) body = body.dropLast(1)
        Link(match.range.first, match.range.first + body.length, body)
    }.toList()
    fun contains(text: String) = find(text).isNotEmpty()
}
