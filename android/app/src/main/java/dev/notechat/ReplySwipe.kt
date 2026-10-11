package dev.notechat

import android.annotation.SuppressLint
import android.content.Context
import android.view.MotionEvent
import android.view.ViewConfiguration
import android.widget.FrameLayout
import kotlin.math.abs

class ReplySwipe @JvmOverloads constructor(context:Context, private val reply:()->Unit={}):FrameLayout(context) {
    var enabledForReply=true
    private var startX=0f
    private var startY=0f
    private var dragging=false
    private val threshold=56*resources.displayMetrics.density
    override fun onInterceptTouchEvent(event:MotionEvent):Boolean {
        if(!enabledForReply)return false
        when(event.actionMasked) {
            MotionEvent.ACTION_DOWN->{ startX=event.x; startY=event.y; dragging=false }
            MotionEvent.ACTION_MOVE->{ val dx=event.x-startX; val dy=event.y-startY
                if(abs(dx)>ViewConfiguration.get(context).scaledTouchSlop*2 && abs(dx)>abs(dy)*1.5) { dragging=true; parent.requestDisallowInterceptTouchEvent(true); return true }
            }
        }
        return dragging
    }
    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(event:MotionEvent):Boolean {
        if(!dragging)return super.onTouchEvent(event)
        val child=getChildAt(0)
        when(event.actionMasked) {
            MotionEvent.ACTION_MOVE->child.translationX=(event.x-startX).coerceIn(-threshold*1.4f,threshold*1.4f)
            MotionEvent.ACTION_UP,MotionEvent.ACTION_CANCEL->{ if(event.actionMasked==MotionEvent.ACTION_UP && abs(child.translationX)>=threshold) { performHapticFeedback(android.view.HapticFeedbackConstants.KEYBOARD_TAP); reply() }; child.animate().translationX(0f).setDuration(160).start(); dragging=false; parent.requestDisallowInterceptTouchEvent(false) }
        }
        return true
    }
}
