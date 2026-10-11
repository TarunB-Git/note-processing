package dev.notechat

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Canvas
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import androidx.appcompat.widget.AppCompatImageView

class ZoomImage(context:Context):AppCompatImageView(context) {
    private var zoom=1f
    private var dx=0f
    private var dy=0f
    private var lastX=0f
    private var lastY=0f
    private val scaling=ScaleGestureDetector(context,object:ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(detector:ScaleGestureDetector):Boolean { scaleTo((zoom*detector.scaleFactor).coerceIn(1f,5f),detector.focusX,detector.focusY); return true }
    })
    private val gestures=GestureDetector(context,object:GestureDetector.SimpleOnGestureListener() {
        override fun onDown(event:MotionEvent)=true
        override fun onDoubleTap(event:MotionEvent):Boolean { scaleTo(if(zoom>1f)1f else 2.5f,event.x,event.y); return true }
        override fun onSingleTapConfirmed(event:MotionEvent)=performClick()
    })
    private fun scaleTo(next:Float,x:Float,y:Float) {
        val ratio=next/zoom
        dx=(dx-(x-width/2f))*ratio+(x-width/2f)
        dy=(dy-(y-height/2f))*ratio+(y-height/2f)
        zoom=next; clamp(); invalidate()
    }
    private fun clamp() {
        val image=drawable ?: return
        val rect=android.graphics.RectF(0f,0f,image.intrinsicWidth.toFloat(),image.intrinsicHeight.toFloat())
        imageMatrix.mapRect(rect)
        val maxX=((rect.width()*zoom-width)/2).coerceAtLeast(0f)
        val maxY=((rect.height()*zoom-height)/2).coerceAtLeast(0f)
        dx=dx.coerceIn(-maxX,maxX); dy=dy.coerceIn(-maxY,maxY)
    }
    override fun onSizeChanged(w:Int,h:Int,oldw:Int,oldh:Int) { super.onSizeChanged(w,h,oldw,oldh); clamp() }
    override fun onDraw(canvas:Canvas) { val saved=canvas.save(); canvas.translate(dx,dy); canvas.scale(zoom,zoom,width/2f,height/2f); super.onDraw(canvas); canvas.restoreToCount(saved) }
    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(event:MotionEvent):Boolean {
        scaling.onTouchEvent(event); gestures.onTouchEvent(event)
        if(event.actionMasked==MotionEvent.ACTION_MOVE && !scaling.isInProgress && zoom>1f) { dx+=event.x-lastX; dy+=event.y-lastY; clamp(); invalidate() }
        lastX=event.x; lastY=event.y; return true
    }
    override fun performClick():Boolean { super.performClick(); return true }
}
