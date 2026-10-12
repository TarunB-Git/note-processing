package dev.notechat

object CategoryIcons {
    val choices = linkedMapOf(
        "thoughts" to R.drawable.ic_category_thoughts,
        "ideas" to R.drawable.ic_category_ideas,
        "todos" to R.drawable.ic_category_todos,
        "questions" to R.drawable.ic_category_questions,
        "links" to R.drawable.ic_category_links,
        "junk" to R.drawable.ic_category_junk,
        "work" to R.drawable.ic_category_work,
        "chat" to R.drawable.ic_category_chat,
        "starred" to R.drawable.ic_category_starred,
        "unsorted" to R.drawable.ic_category_unsorted
    )
    fun resource(id: String, icon: String?) = choices[icon] ?: choices[id] ?: R.drawable.ic_category_chat
}
