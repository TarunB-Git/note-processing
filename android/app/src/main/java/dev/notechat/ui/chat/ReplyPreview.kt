package dev.notechat

import android.content.Context
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.text.TextUtils
import android.view.Gravity
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView

class ReplyPreview(context: Context, message: Message?, colors: UiColors, close: (() -> Unit)? = null) : LinearLayout(context) {
    val summary = MessageText(context)
    val photo: ImageView?
    private fun dp(value: Int) = (value * resources.displayMetrics.density).toInt()

    init {
        orientation = HORIZONTAL
        gravity = Gravity.CENTER_VERTICAL
        background = GradientDrawable().apply { setColor(colors.quote); cornerRadius = dp(6).toFloat() }
        clipToOutline = true
        addView(android.view.View(context).apply { setBackgroundColor(colors.accent) }, LayoutParams(dp(3), -1))
        val text = LinearLayout(context).apply {
            orientation = VERTICAL
            setPadding(dp(10), dp(7), dp(8), dp(7))
        }
        text.addView(TextView(context).apply {
            setText(message?.sender ?: "You")
            setTextColor(colors.accent); textSize = 13f; setTypeface(null, Typeface.BOLD)
            maxLines = 1; ellipsize = TextUtils.TruncateAt.END
        })
        summary.apply {
            setTextColor(colors.ink); textSize = 14f; maxLines = 2
            ellipsize = TextUtils.TruncateAt.END
            setPadding(0, dp(3), 0, 0)
            this.text = message?.text?.ifBlank {
                when {
                    message.attachment?.mime?.startsWith("image/") == true -> "Photo"
                    message.attachment?.mime?.startsWith("audio/") == true -> "Voice message"
                    else -> message.attachment?.name.orEmpty()
                }
            } ?: "Deleted message"
        }
        summary.interceptTap = { this@ReplyPreview.performClick(); true }
        summary.setOnLongClickListener { this@ReplyPreview.performLongClick() }
        text.addView(summary)
        addView(text, LayoutParams(0, -2, 1f))
        photo = if (message?.attachment?.mime?.startsWith("image/") == true) ImageView(context).apply {
            scaleType = ImageView.ScaleType.CENTER_CROP
            tag = message.id; contentDescription = "Reply photo"
            addView(this, LayoutParams(dp(52), dp(52)))
        } else null
        if (close != null) addView(ImageButton(context).apply {
            setImageDrawable(ChatIcon("close", colors.muted)); background = null
            contentDescription = "Cancel reply"; setOnClickListener { close() }
        }, LayoutParams(dp(44), dp(52)))
    }
}
