package dev.notechat

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.view.View
import android.view.animation.LinearInterpolator

class ModelFlowView(context:Context):View(context) {
    private var phase=0f
    private val paint=Paint(Paint.ANTI_ALIAS_FLAG)
    private val animator=ValueAnimator.ofFloat(0f,4f).apply { duration=6400; repeatCount=ValueAnimator.INFINITE; interpolator=LinearInterpolator(); addUpdateListener { phase=it.animatedValue as Float; invalidate() } }
    init { contentDescription="How learning works: message, suggestion, thumbs-up, remembered example. Thumbs-down tries another suggestion without learning." }
    override fun onAttachedToWindow() { super.onAttachedToWindow(); if(ValueAnimator.areAnimatorsEnabled())animator.start() }
    override fun onDetachedFromWindow() { animator.cancel(); super.onDetachedFromWindow() }
    override fun onDraw(canvas:Canvas) {
        val names=listOf("Message","Tag","Approve","Learn")
        val step=width/4f
        val center=height*.38f
        val radius=(step*.25f).coerceAtMost(22*resources.displayMetrics.density)
        paint.color=Color.rgb(213,230,222); paint.strokeWidth=3*resources.displayMetrics.density
        canvas.drawLine(step/2,center,width-step/2,center,paint)
        paint.textAlign=Paint.Align.CENTER; paint.textSize=android.util.TypedValue.applyDimension(android.util.TypedValue.COMPLEX_UNIT_SP,12f,resources.displayMetrics)
        names.forEachIndexed { i,name ->
            val x=step*(i+.5f)
            paint.color=if(phase.toInt()==i)UiColors.of(context).accent else Color.rgb(229,241,235)
            canvas.drawCircle(x,center,radius,paint)
            paint.color=if(phase.toInt()==i)Color.WHITE else Color.rgb(73,115,98)
            canvas.drawText((i+1).toString(),x,center+paint.textSize*.35f,paint)
            paint.color=UiColors.of(context).muted; canvas.drawText(name,x,center+radius+paint.textSize*1.6f,paint)
        }
    }
}
