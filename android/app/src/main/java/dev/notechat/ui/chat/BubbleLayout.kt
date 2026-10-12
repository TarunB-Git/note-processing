package dev.notechat

import android.content.Context
import android.widget.LinearLayout

class BubbleLayout @JvmOverloads constructor(context:Context,private val maximumWidth:Int=context.resources.displayMetrics.widthPixels):LinearLayout(context) {
    override fun onMeasure(widthMeasureSpec:Int,heightMeasureSpec:Int) {
        val width=MeasureSpec.getSize(widthMeasureSpec).coerceAtMost(maximumWidth)
        super.onMeasure(MeasureSpec.makeMeasureSpec(width,MeasureSpec.AT_MOST),heightMeasureSpec)
    }
}
