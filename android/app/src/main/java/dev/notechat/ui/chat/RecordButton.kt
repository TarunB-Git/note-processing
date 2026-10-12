package dev.notechat

import android.annotation.SuppressLint
import android.content.Context
import android.os.Handler
import android.os.Looper
import android.view.MotionEvent
import androidx.appcompat.widget.AppCompatImageButton

class RecordButton(context:Context):AppCompatImageButton(context) {
    var startHold:()->Boolean={false}
    var releaseHold:()->Unit={}
    var cancelHold:()->Unit={}
    var lockHold:()->Unit={}
    var gesturesEnabled=true
    private val handler=Handler(Looper.getMainLooper())
    private var held=false
    private var locked=false
    private var cancelled=false
    private var x=0f
    private var y=0f
    private val start=Runnable { held=startHold(); if(held) { performHapticFeedback(android.view.HapticFeedbackConstants.LONG_PRESS); parent.requestDisallowInterceptTouchEvent(true) } }
    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(event:MotionEvent):Boolean {
        if(!gesturesEnabled && !held)return super.onTouchEvent(event)
        when(event.actionMasked) {
            MotionEvent.ACTION_DOWN->{ x=event.rawX; y=event.rawY; held=false; locked=false; cancelled=false; handler.postDelayed(start,300); return true }
            MotionEvent.ACTION_MOVE->if(held && !locked) {
                if(event.rawX-x < -100*resources.displayMetrics.density) { handler.removeCallbacks(start); held=false; cancelled=true; cancelHold(); return true }
                if(event.rawY-y < -65*resources.displayMetrics.density) { locked=true; lockHold() }
            }
            MotionEvent.ACTION_UP->{ handler.removeCallbacks(start); if(held) { if(!locked)releaseHold() } else if(!cancelled)performClick(); parent.requestDisallowInterceptTouchEvent(false); held=false; return true }
            MotionEvent.ACTION_CANCEL->{ handler.removeCallbacks(start); if(held && !locked)cancelHold(); held=false; return true }
        }
        return true
    }
    override fun performClick():Boolean { super.performClick(); return true }
    override fun onDetachedFromWindow() { handler.removeCallbacks(start); super.onDetachedFromWindow() }
}
