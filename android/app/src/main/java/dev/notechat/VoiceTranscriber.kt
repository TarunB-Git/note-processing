package dev.notechat

import android.content.Context
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import org.json.JSONObject
import org.vosk.Model
import org.vosk.Recognizer
import java.io.File
import java.net.URL
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.zip.ZipInputStream

object VoiceTranscriber {
    private const val MODEL="vosk-model-small-en-us-0.15"
    private const val CHECKSUM="30f26242c4eb449f948e42cb302dd7a686cb29a3423a8367f99ff41780942498"
    private fun directory(context:Context)=File(context.filesDir,"speech-vosk")
    fun installed(context:Context)=File(directory(context),"$MODEL/am/final.mdl").isFile
    fun remove(context:Context) { check(directory(context).deleteRecursively()) { "Could not remove speech model" } }
    fun download(context:Context,progress:(Int)->Unit) {
        val archive=File(context.cacheDir,"speech-model.zip.part")
        val stage=File(context.filesDir,"speech-vosk.part")
        stage.deleteRecursively(); check(stage.mkdirs())
        val connection=URL("https://alphacephei.com/vosk/models/$MODEL.zip").openConnection().apply { connectTimeout=30_000; readTimeout=30_000 }
        try {
            val digest=MessageDigest.getInstance("SHA-256")
            connection.getInputStream().use { input -> archive.outputStream().use { output ->
                val buffer=ByteArray(65536); var total=0L
                while(true) {
                    val count=input.read(buffer); if(count<0)break
                    total+=count; require(total<=50L*1024*1024) { "Unexpected speech model size" }
                    digest.update(buffer,0,count); output.write(buffer,0,count)
                    progress((total*100/connection.contentLengthLong.coerceAtLeast(1)).toInt().coerceIn(0,100))
                }
            } }
            require(digest.digest().joinToString("") { "%02x".format(it) }==CHECKSUM) { "Speech model checksum failed" }
            unpack(archive,stage)
            require(File(stage,"$MODEL/am/final.mdl").isFile && File(stage,"$MODEL/conf/mfcc.conf").isFile) { "Incomplete speech model" }
            check(directory(context).deleteRecursively()); check(stage.renameTo(directory(context))) { "Could not install speech model" }
        } finally { archive.delete(); stage.deleteRecursively() }
    }
    internal fun unpack(archive:File,stage:File) {
        val root=stage.canonicalPath+File.separator
        var total=0L; var entries=0
        ZipInputStream(archive.inputStream()).use { input ->
            while(true) {
                val entry=input.nextEntry ?: break
                require(++entries<=500) { "Too many model files" }
                val file=File(stage,entry.name)
                require(file.canonicalPath.startsWith(root)) { "Unsafe model path" }
                if(entry.isDirectory)file.mkdirs() else {
                    file.parentFile?.mkdirs()
                    file.outputStream().use { output ->
                        val buffer=ByteArray(65536)
                        while(true) { val count=input.read(buffer); if(count<0)break; total+=count; require(total<=128L*1024*1024) { "Expanded model is too large" }; output.write(buffer,0,count) }
                    }
                }
                input.closeEntry()
            }
        }
    }
    fun transcribe(context:Context,file:File,cancelled:()->Boolean):String {
        require(installed(context)) { "Download the offline speech model first" }
        val extractor=MediaExtractor()
        var codec:MediaCodec?=null
        var recognizer:Recognizer?=null
        val model=Model(File(directory(context),MODEL).absolutePath)
        val segments=mutableListOf<String>()
        fun result(value:String) { JSONObject(value).optString("text").takeIf { it.isNotBlank() }?.let { segments.add(it) } }
        try {
            extractor.setDataSource(file.absolutePath)
            val track=(0 until extractor.trackCount).firstOrNull { extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("audio/")==true } ?: error("No audio track")
            extractor.selectTrack(track)
            val format=extractor.getTrackFormat(track)
            require(!format.containsKey(MediaFormat.KEY_DURATION) || format.getLong(MediaFormat.KEY_DURATION)<=10L*60*1_000_000) { "Transcribe recordings up to 10 minutes" }
            codec=MediaCodec.createDecoderByType(format.getString(MediaFormat.KEY_MIME)!!)
            codec.configure(format,null,null,0); codec.start()
            var inputEnded=false; var outputEnded=false
            var rate=format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            var channels=format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
            var encoding=AudioFormat.ENCODING_PCM_16BIT
            val info=MediaCodec.BufferInfo()
            val deadline=android.os.SystemClock.elapsedRealtime()+15*60*1000L
            while(!outputEnded) {
                check(!cancelled()) { "Transcription cancelled" }
                check(android.os.SystemClock.elapsedRealtime()<deadline) { "Transcription timed out" }
                if(!inputEnded) {
                    val index=codec.dequeueInputBuffer(10_000)
                    if(index>=0) {
                        val buffer=codec.getInputBuffer(index)!!; val count=extractor.readSampleData(buffer,0)
                        if(count<0) { codec.queueInputBuffer(index,0,0,0,MediaCodec.BUFFER_FLAG_END_OF_STREAM); inputEnded=true }
                        else { codec.queueInputBuffer(index,0,count,extractor.sampleTime,0); extractor.advance() }
                    }
                }
                when(val index=codec.dequeueOutputBuffer(info,10_000)) {
                    MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        val output=codec.outputFormat; rate=output.getInteger(MediaFormat.KEY_SAMPLE_RATE); channels=output.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                        encoding=if(output.containsKey(MediaFormat.KEY_PCM_ENCODING))output.getInteger(MediaFormat.KEY_PCM_ENCODING) else AudioFormat.ENCODING_PCM_16BIT
                        require(encoding==AudioFormat.ENCODING_PCM_16BIT || encoding==AudioFormat.ENCODING_PCM_FLOAT) { "Unsupported decoded audio format" }
                    }
                    else -> if(index>=0) {
                        val buffer=codec.getOutputBuffer(index)!!.apply { position(info.offset); limit(info.offset+info.size); order(ByteOrder.LITTLE_ENDIAN) }
                        if(info.size>0) {
                            if(recognizer==null)recognizer=Recognizer(model,rate.toFloat())
                            val mono=monoPcm(buffer,channels,encoding==AudioFormat.ENCODING_PCM_FLOAT)
                            if(recognizer!!.acceptWaveForm(mono,mono.size))result(recognizer!!.result)
                        }
                        outputEnded=info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM!=0
                        codec.releaseOutputBuffer(index,false)
                    }
                }
            }
            recognizer?.let { result(it.finalResult) }
            return segments.joinToString(" ").also { require(it.isNotBlank()) { "No speech recognized. The English model may not match this recording." } }
        } finally {
            recognizer?.close(); model.close()
            try { codec?.stop() } catch(_:Exception) { }; codec?.release(); extractor.release()
        }
    }
    internal fun monoPcm(buffer:java.nio.ByteBuffer,channels:Int,floating:Boolean):FloatArray {
        require(channels in 1..8)
        val frames=buffer.remaining()/(if(floating)4 else 2)/channels
        return FloatArray(frames) {
            var sum=0f
            repeat(channels) { sum+=if(floating)buffer.float.coerceIn(-1f,1f)*32768f else buffer.short.toFloat() }
            sum/channels
        }
    }
}
