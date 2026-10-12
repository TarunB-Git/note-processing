package dev.notechat

import android.content.Context
import android.graphics.Color
import android.view.Gravity
import android.widget.*

class VoiceMessageView @JvmOverloads constructor(context:Context, waveform:List<Int> = emptyList(), private val toggle:()->Unit={}, seek:(Int)->Unit={}, speed:()->Unit={}):LinearLayout(context) {
    private val play=ImageButton(context)
    private val wave=WaveformView(context).apply { samples=waveform }
    private val slider=SeekBar(context)
    private val clock=TextView(context)
    private val rate=TextView(context)
    var duration=0
    var selectionMode=false
        set(value) { field=value; play.isEnabled=!value; slider.isEnabled=!value; rate.isEnabled=!value }
    init {
        orientation=HORIZONTAL; gravity=Gravity.CENTER_VERTICAL
        setOnClickListener { if(selectionMode)toggle() }
        play.apply { background=null; setPadding(dp(10),dp(10),dp(10),dp(10)); contentDescription="Play voice message"; setOnClickListener { toggle() } }
        addView(play,LayoutParams(dp(48),dp(48)))
        val body=LinearLayout(context).apply { orientation=VERTICAL }
        val track=FrameLayout(context)
        track.addView(wave,FrameLayout.LayoutParams(-1,dp(32)).apply { leftMargin=dp(8); rightMargin=dp(8) })
        slider.apply { max=1000; progressTintList=android.content.res.ColorStateList.valueOf(Color.TRANSPARENT); progressBackgroundTintList=android.content.res.ColorStateList.valueOf(Color.TRANSPARENT); thumbTintList=android.content.res.ColorStateList.valueOf(UiColors.of(context).accent); setPadding(dp(8),0,dp(8),0); contentDescription="Seek voice message" }
        track.addView(slider,FrameLayout.LayoutParams(-1,dp(32)))
        slider.setOnSeekBarChangeListener(object:SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(bar:SeekBar?,value:Int,fromUser:Boolean) { if(fromUser) { wave.progress=value/1000f; seek((duration*value/1000L).toInt()) } }
            override fun onStartTrackingTouch(bar:SeekBar?)=Unit
            override fun onStopTrackingTouch(bar:SeekBar?)=Unit
        })
        body.addView(track)
        val footer=LinearLayout(context).apply { gravity=Gravity.CENTER_VERTICAL }
        clock.apply { textSize=11f; setTextColor(UiColors.of(context).muted) }
        footer.addView(clock,LayoutParams(0,-2,1f))
        rate.apply { textSize=12f; setTextColor(UiColors.of(context).accent); gravity=Gravity.CENTER; minWidth=dp(44); minHeight=dp(32); contentDescription="Change playback speed"; setOnClickListener { speed() } }
        footer.addView(rate); body.addView(footer)
        addView(body,LayoutParams(0,-2,1f)); update(false,0,0,1f)
    }
    fun update(playing:Boolean,position:Int,total:Int,speed:Float,loading:Boolean=false) {
        duration=total
        play.isEnabled=!loading && !selectionMode
        play.setImageDrawable(ChatIcon(if(playing)"pause" else "play",UiColors.of(context).muted))
        play.contentDescription=if(playing)"Pause voice message" else "Play voice message"
        if(!slider.isPressed)slider.progress=if(total>0)(position*1000L/total).toInt() else 0
        wave.progress=if(total>0)position.toFloat()/total else 0f
        clock.text=if(loading)"Loading…" else if(position>0)"${duration(position)} / ${duration(total)}" else duration(total)
        rate.text="${if(speed==1f)"1" else speed.toString().removeSuffix(".0")}×"
    }
    private fun duration(ms:Int)="${ms/60000}:${(ms/1000%60).toString().padStart(2,'0')}"
    private fun dp(value:Int)=(value*resources.displayMetrics.density).toInt()
}
