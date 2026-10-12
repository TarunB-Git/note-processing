package dev.notechat

import android.content.Context
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import kotlin.math.abs

/** Horizontal page gestures yield to the horizontally scrollable board previews. */
class HomeSwipe(context: Context, private val switchTab: (Int) -> Unit) : FrameLayout(context) {
    private var x=0f
    private var y=0f
    private var captured=false
    private var preview=false
    private fun hitsPreview(view: View,event: MotionEvent): Boolean {
        if(view is RecyclerView && (view.layoutManager as? LinearLayoutManager)?.orientation==RecyclerView.HORIZONTAL) {
            val bounds=android.graphics.Rect()
            if(view.getGlobalVisibleRect(bounds) && bounds.contains(event.rawX.toInt(),event.rawY.toInt()))return true
        }
        return view is ViewGroup && (0 until view.childCount).any { hitsPreview(view.getChildAt(it),event) }
    }
    override fun onInterceptTouchEvent(event: MotionEvent): Boolean {
        if(event.actionMasked==MotionEvent.ACTION_DOWN) { x=event.x; y=event.y; captured=false; preview=hitsPreview(this,event) }
        if(!preview && event.actionMasked==MotionEvent.ACTION_MOVE && abs(event.x-x)>64*resources.displayMetrics.density && abs(event.y-y)<32*resources.displayMetrics.density)captured=true
        return captured
    }
    override fun onTouchEvent(event: MotionEvent): Boolean {
        if(!captured)return super.onTouchEvent(event)
        if(event.actionMasked==MotionEvent.ACTION_UP) { captured=false; performClick(); switchTab(if(event.x<x)1 else -1) }
        if(event.actionMasked==MotionEvent.ACTION_CANCEL)captured=false
        return true
    }
    override fun performClick(): Boolean = super.performClick()
}
