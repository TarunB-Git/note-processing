package dev.notechat

import android.graphics.Canvas
import android.graphics.ColorFilter
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PixelFormat
import android.graphics.drawable.Drawable

class ChatIcon(private val name: String, private val color: Int) : Drawable() {
    override fun draw(canvas: Canvas) {
        val save = canvas.save()
        val scale = minOf(bounds.width(), bounds.height()) / 24f
        canvas.translate(bounds.left + (bounds.width() - 24 * scale) / 2, bounds.top + (bounds.height() - 24 * scale) / 2)
        canvas.scale(scale, scale)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = this@ChatIcon.color; style = Paint.Style.STROKE; strokeWidth = 1.8f; strokeCap = Paint.Cap.ROUND; strokeJoin = Paint.Join.ROUND }
        fun line(vararg points: Float) {
            val path = Path().apply { moveTo(points[0], points[1]); for (i in 2 until points.size step 2) lineTo(points[i], points[i + 1]) }
            canvas.drawPath(path, paint)
        }
        when (name) {
            "back" -> { line(20f,12f,4f,12f); line(11f,5f,4f,12f,11f,19f) }
            "send" -> { line(3f,3f,21f,12f,3f,21f,6f,12f,3f,3f); line(6f,12f,21f,12f) }
            "search" -> { canvas.drawCircle(10f,10f,6f,paint); line(15f,15f,21f,21f) }
            "more" -> { paint.style=Paint.Style.FILL; listOf(5f,12f,19f).forEach { canvas.drawCircle(12f,it,1.5f,paint) } }
            "mic" -> { canvas.drawRoundRect(8f,3f,16f,14f,4f,4f,paint); canvas.drawArc(5f,6f,19f,19f,0f,180f,false,paint); line(12f,19f,12f,22f,8f,22f,16f,22f) }
            "attach" -> { val path=Path().apply { moveTo(8f,14f); lineTo(16f,6f); cubicTo(20f,2f,25f,7f,21f,11f); lineTo(11f,21f); cubicTo(6f,26f,0f,20f,5f,15f); lineTo(15f,5f) }; canvas.drawPath(path,paint) }
            "camera" -> { canvas.drawRoundRect(3f,6f,21f,21f,2f,2f,paint); line(7f,6f,9f,3f,15f,3f,17f,6f); canvas.drawCircle(12f,13f,4f,paint) }
            "chat" -> { canvas.drawRoundRect(3f,3f,21f,17f,3f,3f,paint); line(6f,17f,6f,22f,12f,17f); line(7f,8f,17f,8f); line(7f,12f,14f,12f) }
            "add" -> { line(12f,4f,12f,20f); line(4f,12f,20f,12f) }
            "check" -> line(4f,12f,9f,17f,20f,6f)
            "close" -> { line(5f,5f,19f,19f); line(19f,5f,5f,19f) }
            "play" -> line(8f,4f,20f,12f,8f,20f,8f,4f)
            "stop" -> canvas.drawRect(5f,5f,19f,19f,paint)
            "pause" -> { line(8f,5f,8f,19f); line(16f,5f,16f,19f) }
            "up" -> line(5f,15f,12f,8f,19f,15f)
            "down" -> line(5f,9f,12f,16f,19f,9f)
            "latest" -> { line(5f,6f,12f,13f,19f,6f); line(5f,12f,12f,19f,19f,12f) }
            "reply" -> { line(10f,5f,3f,12f,10f,19f); line(3f,12f,14f,12f); canvas.drawArc(10f,12f,22f,24f,270f,90f,false,paint) }
            "forward" -> { line(14f,4f,22f,11f,14f,18f); line(22f,11f,12f,11f); canvas.drawArc(2f,11f,14f,23f,180f,90f,false,paint) }
            "edit" -> { line(4f,16f,16f,4f,20f,8f,8f,20f,4f,20f,4f,16f); line(14f,6f,18f,10f) }
            "notes" -> { canvas.drawRoundRect(4f,2f,20f,22f,2f,2f,paint); line(8f,7f,16f,7f); line(8f,12f,16f,12f); line(8f,17f,13f,17f) }
            "copy" -> { canvas.drawRoundRect(8f,7f,21f,22f,1f,1f,paint); line(16f,3f,3f,3f,3f,17f) }
            "delete" -> { line(4f,6f,20f,6f); line(9f,3f,15f,3f); line(6f,6f,7f,21f,17f,21f,18f,6f); line(10f,10f,10f,17f); line(14f,10f,14f,17f) }
            "share" -> { canvas.drawCircle(5f,12f,2.5f,paint); canvas.drawCircle(19f,5f,2.5f,paint); canvas.drawCircle(19f,19f,2.5f,paint); line(7f,11f,17f,6f); line(7f,13f,17f,18f) }
            "settings" -> { canvas.drawCircle(12f,12f,7f,paint); canvas.drawCircle(12f,12f,2.5f,paint); for(i in 0..7) { val angle=i*Math.PI/4; line((12+7*kotlin.math.cos(angle)).toFloat(),(12+7*kotlin.math.sin(angle)).toFloat(),(12+10*kotlin.math.cos(angle)).toFloat(),(12+10*kotlin.math.sin(angle)).toFloat()) } }
            "pin" -> { line(8f,3f,16f,3f,15f,10f,19f,15f,5f,15f,9f,10f,8f,3f); line(12f,15f,12f,22f) }
            "star" -> line(12f,2f,15f,8f,22f,9f,17f,14f,18f,22f,12f,18f,6f,22f,7f,14f,2f,9f,9f,8f,12f,2f)
            "lock" -> { canvas.drawRoundRect(4f,10f,20f,22f,2f,2f,paint); canvas.drawArc(7f,2f,17f,16f,180f,180f,false,paint); line(12f,15f,12f,18f) }
        }
        canvas.restoreToCount(save)
    }
    override fun setAlpha(alpha: Int) = Unit
    override fun setColorFilter(colorFilter: ColorFilter?) = Unit
    @Deprecated("Deprecated in Java") override fun getOpacity() = PixelFormat.TRANSLUCENT
}
