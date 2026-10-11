package dev.notechat

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.drawable.ColorDrawable
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.RippleDrawable
import android.content.res.ColorStateList
import android.view.Gravity
import android.view.View
import android.widget.FrameLayout
import android.widget.ImageButton
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView

class ChatViewport(context: Context, val recycler: RecyclerView, private val ids: List<String>, private val colors: UiColors) : FrameLayout(context) {
    private var focusedId: String? = null
    private var focusUntil = 0L
    val latest = ImageButton(context)
    private fun dp(value: Int) = (value * resources.displayMetrics.density).toInt()

    init {
        addView(recycler, LayoutParams(-1, -1))
        latest.apply {
            contentDescription = "Go to latest message"
            setImageDrawable(ChatIcon("latest", colors.muted))
            val circle = GradientDrawable().apply { shape = GradientDrawable.OVAL; setColor(this@ChatViewport.colors.surface) }
            background = RippleDrawable(ColorStateList.valueOf(colors.selected), circle, null)
            elevation = dp(4).toFloat(); setPadding(dp(10), dp(10), dp(10), dp(10))
            visibility = View.GONE
            setOnClickListener {
                val manager = recycler.layoutManager as LinearLayoutManager
                if (ids.lastIndex - manager.findLastVisibleItemPosition() > 20) manager.scrollToPosition((ids.lastIndex - 8).coerceAtLeast(0))
                recycler.post { recycler.smoothScrollToPosition(ids.lastIndex) }
            }
        }
        addView(latest, LayoutParams(dp(44), dp(44), Gravity.END or Gravity.BOTTOM).apply {
            rightMargin = dp(12); bottomMargin = dp(12)
        })
        recycler.addOnScrollListener(object : RecyclerView.OnScrollListener() {
            override fun onScrolled(view: RecyclerView, dx: Int, dy: Int) { updateLatest() }
        })
        recycler.addOnLayoutChangeListener { _, _, _, _, _, _, _, _, _ -> updateLatest() }
    }
    fun updateLatest() {
        latest.visibility = if (ids.isNotEmpty() && recycler.canScrollVertically(1)) View.VISIBLE else View.GONE
    }
    fun reveal(id: String, highlight: Boolean) {
        val index = ids.indexOf(id)
        if (index < 0) return
        if (highlight) { focusedId = id; focusUntil = android.os.SystemClock.uptimeMillis() + 1300 }
        (recycler.layoutManager as LinearLayoutManager).scrollToPositionWithOffset(index, dp(56))
        recycler.post {
            recycler.findViewHolderForAdapterPosition(index)?.itemView?.let { bind(it, id) }
                ?: recycler.adapter?.notifyItemChanged(index)
        }
    }
    fun bind(view: View, id: String) {
        view.tag = id; view.foreground = null
        if (id != focusedId || android.os.SystemClock.uptimeMillis() >= focusUntil) return
        val shade = ColorDrawable(colors.selected).apply { alpha = 110 }
        view.foreground = shade
        view.announceForAccessibility("Replied message")
        if (ValueAnimator.areAnimatorsEnabled()) ValueAnimator.ofInt(110, 0).apply {
            duration = 1000; startDelay = 120
            addUpdateListener { if (view.tag == id && view.foreground === shade) shade.alpha = it.animatedValue as Int else cancel() }
            start()
        } else view.postDelayed({ if (view.foreground === shade) view.foreground = null }, 1100)
    }
}
