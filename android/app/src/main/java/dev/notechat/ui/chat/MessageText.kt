package dev.notechat

import android.annotation.SuppressLint
import android.content.Context
import android.text.Spanned
import android.text.style.ClickableSpan
import android.view.GestureDetector
import android.view.MotionEvent
import androidx.appcompat.widget.AppCompatTextView

class MessageText @JvmOverloads constructor(context: Context, private val doubleTap: () -> Unit = {}) : AppCompatTextView(context) {
    var interceptTap: (() -> Boolean)? = null
    var linkMenu: ((String) -> Unit)? = null
    private var touchX = 0f
    private var touchY = 0f
    private val listener = object : GestureDetector.SimpleOnGestureListener() {
        override fun onDown(event: MotionEvent): Boolean { touchX=event.x; touchY=event.y; return true }
        override fun onSingleTapUp(event: MotionEvent): Boolean { if(immediateTap)dispatchTap(); return immediateTap }
        override fun onSingleTapConfirmed(event: MotionEvent): Boolean { if(!immediateTap)dispatchTap(); return true }
        override fun onDoubleTap(event: MotionEvent): Boolean { if(interceptTap?.invoke()!=true)doubleTap(); return true }
        override fun onLongPress(event: MotionEvent) {
            val span=linkAtPoint()
            if(span is android.text.style.URLSpan && linkMenu!=null)linkMenu?.invoke(span.url) else performLongClick()
        }
    }
    private val gestures = GestureDetector(context,listener)
    var immediateTap=false
        set(value) { field=value; gestures.setOnDoubleTapListener(if(value)null else listener) }
    private fun dispatchTap() { if(interceptTap?.invoke()!=true)linkAtPoint()?.onClick(this) ?: performClick() }
    private fun linkAtPoint():ClickableSpan? {
        val spans=text as? Spanned ?: return null
        val layout=layout ?: return null
        val x=touchX-totalPaddingLeft+scrollX
        val y=(touchY-totalPaddingTop+scrollY).toInt()
        if(y<0 || y>layout.height)return null
        val line=layout.getLineForVertical(y)
        if(x<layout.getLineLeft(line) || x>layout.getLineRight(line))return null
        val offset=layout.getOffsetForHorizontal(line,x)
        return spans.getSpans(offset,offset,ClickableSpan::class.java).firstOrNull()
    }
    // Confirmed taps reach performClick after the double-tap timeout.
    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(event: MotionEvent): Boolean = gestures.onTouchEvent(event)
    override fun performClick(): Boolean { super.performClick(); return true }
}
