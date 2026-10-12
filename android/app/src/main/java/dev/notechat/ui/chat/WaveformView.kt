package dev.notechat

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.view.View

class WaveformView(context:Context):View(context) {
    var samples:List<Int> = emptyList(); set(value) { field=value; invalidate() }
    var progress=0f; set(value) { field=value.coerceIn(0f,1f); invalidate() }
    var live=false
    private val paint=Paint(Paint.ANTI_ALIAS_FLAG).apply { strokeCap=Paint.Cap.ROUND }
    override fun onDraw(canvas:Canvas) {
        super.onDraw(canvas)
        val count=32
        val spacing=width.toFloat()/count
        paint.strokeWidth=(spacing*.42f).coerceAtLeast(2f)
        val peak=samples.maxOrNull()?.coerceAtLeast(1) ?: 1
        for(i in 0 until count) {
            val index=if(samples.isEmpty())0 else (i*samples.size/count).coerceAtMost(samples.lastIndex)
            val amplitude=if(samples.isEmpty()) .12f else (samples[index].toFloat()/peak).coerceAtLeast(.08f)
            val size=amplitude*height*.7f
            paint.color=if(live) Color.rgb(220,67,80) else if(i.toFloat()/count<progress)UiColors.of(context).accent else Color.rgb(151,177,162)
            canvas.drawLine(spacing*(i+.5f),height/2f-size/2,spacing*(i+.5f),height/2f+size/2,paint)
        }
    }
}
