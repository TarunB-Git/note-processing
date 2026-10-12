package dev.notechat

import android.content.Context
import android.graphics.Color
import android.util.TypedValue

data class UiColors(val dark: Boolean, val notes: Boolean = false) {
    val surface = if(notes && dark)Color.rgb(24,24,27) else if (dark) Color.rgb(17,27,33) else Color.WHITE
    val paper = if(notes)if(dark)Color.rgb(12,12,14) else Color.rgb(246,246,248) else if (dark) Color.rgb(11,20,26) else Color.rgb(239,234,226)
    val ink = if (dark) Color.rgb(233,237,239) else Color.rgb(17,27,33)
    val muted = if (dark) Color.rgb(166,180,189) else Color.rgb(102,119,129)
    val accent = if(notes)Color.rgb(222,96,51) else if (dark) Color.rgb(37,211,166) else Color.rgb(0,128,105)
    val bubble = if (dark) Color.rgb(0,92,75) else Color.rgb(217,253,211)
    val inset = if (dark) Color.rgb(32,44,51) else Color.rgb(244,245,246)
    val quote = if (dark) Color.rgb(0, 70, 58) else Color.rgb(201,236,196)
    val selected = if (dark) Color.rgb(42, 70, 75) else Color.rgb(193,223,217)
    val divider = if (dark) Color.rgb(42,55,63) else Color.rgb(235,238,239)
    companion object {
        fun of(context: Context): UiColors {
            val value = TypedValue()
            context.theme.resolveAttribute(android.R.attr.colorBackground, value, true)
            return UiColors(Color.red(value.data) < 100)
        }
    }
}
