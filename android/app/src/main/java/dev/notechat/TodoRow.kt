package dev.notechat

import android.content.Context
import android.graphics.Paint
import android.view.Gravity
import android.widget.CheckBox
import android.widget.LinearLayout

class TodoRow(context: Context, message: Message, colors: UiColors, markdown: MarkdownText,
              toggle: () -> Unit, select: () -> Unit) : LinearLayout(context) {
    val text = MessageText(context)
    val footer = LinearLayout(context)
    val check = CheckBox(context)
    private fun dp(value: Int) = (value * resources.displayMetrics.density).toInt()

    init {
        orientation = HORIZONTAL
        gravity = Gravity.CENTER_VERTICAL
        setPadding(dp(8), dp(4), dp(16), dp(4))
        check.apply {
            isChecked = message.done; minimumWidth = dp(52); minimumHeight = dp(52)
            scaleX = 1.15f; scaleY = 1.15f
            buttonTintList = android.content.res.ColorStateList.valueOf(colors.accent)
            contentDescription = "Mark task ${if (message.done) "incomplete" else "complete"}"
            setOnClickListener { isChecked = message.done; toggle() }
        }
        addView(check, LayoutParams(dp(56), dp(56)))
        val content = LinearLayout(context).apply { orientation = VERTICAL }
        text.apply {
            textSize = 17f; setTextColor(if (message.done) colors.muted else colors.ink)
            markdown.render(this, message.text.ifBlank { message.attachment?.name.orEmpty() }, "") {}
            if (message.done) paintFlags = paintFlags or Paint.STRIKE_THRU_TEXT_FLAG
            interceptTap = { toggle(); true }
            setOnClickListener { toggle() }
            setOnLongClickListener { select(); true }
        }
        content.addView(text); content.addView(footer)
        content.setOnClickListener { toggle() }
        content.setOnLongClickListener { select(); true }
        addView(content, LayoutParams(0, -2, 1f))
        setOnClickListener { toggle() }
        setOnLongClickListener { select(); true }
    }
}
