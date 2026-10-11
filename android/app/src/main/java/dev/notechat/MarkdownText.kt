package dev.notechat

import android.content.Context
import android.graphics.Color
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.TextPaint
import android.text.style.BackgroundColorSpan
import android.text.style.ClickableSpan
import android.text.style.URLSpan
import io.noties.markwon.Markwon
import io.noties.markwon.AbstractMarkwonPlugin
import io.noties.markwon.MarkwonConfiguration
import io.noties.markwon.core.MarkwonTheme
import io.noties.markwon.ext.strikethrough.StrikethroughPlugin
import io.noties.markwon.ext.tables.TablePlugin
import io.noties.markwon.ext.tasklist.TaskListPlugin

class MarkdownText(context:Context, open:(String)->Unit) {
    private val blue=if(UiColors.of(context).dark)Color.rgb(83,189,235) else Color.rgb(2,119,189)
    private val renderer=Markwon.builder(context)
        .usePlugin(StrikethroughPlugin.create()).usePlugin(TablePlugin.create(context)).usePlugin(TaskListPlugin.create(context))
        .usePlugin(object:AbstractMarkwonPlugin() {
            override fun configureTheme(builder:MarkwonTheme.Builder) { builder.linkColor(blue) }
            override fun configureConfiguration(builder:MarkwonConfiguration.Builder) { builder.linkResolver { _,link->open(link) } }
        }).build()
    fun render(view:MessageText, source:String, query:String, open:(String)->Unit) {
        val output=SpannableStringBuilder(renderer.toMarkdown(source))
        for(link in TextLinks.find(output.toString())) {
            val existing=output.getSpans(link.start,link.end,ClickableSpan::class.java)
            val code=output.getSpans(link.start,link.end,Any::class.java).any { it.javaClass.simpleName.contains("Code") }
            if(existing.isEmpty() && !code) output.setSpan(object:URLSpan(link.url) {
                override fun onClick(widget:android.view.View) { open(url) }
                override fun updateDrawState(paint:TextPaint) { paint.color=blue; paint.isUnderlineText=false }
            },link.start,link.end,Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        }
        if(query.isNotBlank()) {
            var start=output.toString().indexOf(query,ignoreCase=true)
            while(start>=0) { output.setSpan(BackgroundColorSpan(Color.rgb(255,225,108)),start,start+query.length,Spanned.SPAN_EXCLUSIVE_EXCLUSIVE); start=output.toString().indexOf(query,start+query.length,true) }
        }
        renderer.setParsedMarkdown(view,output)
        view.setLinkTextColor(blue)
    }
}
