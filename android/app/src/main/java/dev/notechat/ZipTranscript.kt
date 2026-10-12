package dev.notechat

import java.io.File
import java.util.zip.ZipFile

object ZipTranscript {
    data class Text(val name: String, val source: String, val messages: Int)
    fun read(file: File): List<Text> = ZipFile(file).use { zip ->
        val entries=zip.entries().asSequence().take(5001).toList()
        require(entries.size<=5000) { "ZIP contains too many entries" }
        var total=0
        entries.filter { !it.isDirectory && (it.name.endsWith(".txt",true) || it.name.endsWith(".md",true)) }.map { entry ->
            val text=zip.getInputStream(entry).use { input ->
                val out=java.io.ByteArrayOutputStream(); val buffer=ByteArray(8192)
                while(true) { val count=input.read(buffer); if(count<0)break; total+=count; require(out.size()+count<=10*1024*1024 && total<=20*1024*1024) { "ZIP text exceeds the import limit" }; out.write(buffer,0,count) }
                out.toString("UTF-8")
            }
            Text(entry.name,text,ChatImport.parse(text,true).entries.size)
        }.filter { it.messages>0 }.sortedByDescending { if(ChatImport.parse(it.source,true).transcript)100000+it.messages else it.messages }
    }
}
