package dev.notechat

import android.content.Context
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.view.Gravity
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.TextView

class GuideCoach(
    context: Context, progress: GuideProgress, colors: UiColors,
    previous: ()->Unit, show: ()->Unit, next: ()->Unit, pause: ()->Unit
) : LinearLayout(context) {
    private fun dp(value: Int)=(value*resources.displayMetrics.density).toInt()
    init {
        val step=GuidePages.steps[progress.step]
        orientation=VERTICAL
        setPadding(dp(14),dp(8),dp(14),dp(6))
        background=GradientDrawable().apply {
            setColor(colors.surface)
            setStroke(dp(1),colors.accent)
            cornerRadius=dp(16).toFloat()
        }
        contentDescription="Guide step ${progress.step+1}: ${step.title}"
        val header=LinearLayout(context).apply { gravity=Gravity.CENTER_VERTICAL }
        header.addView(TextView(context).apply {
            text="${progress.step+1}/${GuidePages.steps.size}  ${step.title}"
            textSize=14f
            setTextColor(colors.ink)
            setTypeface(null,Typeface.BOLD)
        },LayoutParams(0,-2,1f))
        header.addView(ImageButton(context).apply {
            contentDescription="Pause guide"
            background=null
            setImageDrawable(ChatIcon("close",colors.muted))
            setPadding(dp(10),dp(10),dp(10),dp(10))
            setOnClickListener { pause() }
        },LayoutParams(dp(40),dp(40)))
        addView(header)
        addView(TextView(context).apply {
            text=step.text
            textSize=13f
            setTextColor(colors.muted)
            setPadding(0,0,0,dp(4))
        })
        val controls=LinearLayout(context).apply { gravity=Gravity.END }
        fun button(title: String, enabled: Boolean=true, click: ()->Unit) = TextView(context).apply {
            text=title
            textSize=13f
            gravity=Gravity.CENTER
            minHeight=dp(40)
            setPadding(dp(12),dp(6),dp(12),dp(6))
            setTextColor(colors.accent)
            isEnabled=enabled
            alpha=if(enabled)1f else .35f
            contentDescription="Guide $title"
            setOnClickListener { click() }
        }
        controls.addView(button("Previous",progress.step>0,previous))
        controls.addView(button("Show",click=show))
        controls.addView(button(if(progress.step==GuidePages.steps.lastIndex)"Finish" else "Next",click=next))
        addView(controls)
    }
}
