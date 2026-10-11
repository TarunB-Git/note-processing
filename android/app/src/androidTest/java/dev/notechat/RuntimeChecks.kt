package dev.notechat

import android.app.Activity
import android.app.Instrumentation
import android.content.ContextWrapper
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.os.Bundle
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import java.io.ByteArrayInputStream
import java.io.File
import java.security.KeyStore
import java.util.UUID

class RuntimeChecks : Instrumentation() {
    override fun onCreate(arguments: Bundle?) { super.onCreate(arguments); start() }
    override fun onStart() {
        val output=Bundle()
        val directory=File(targetContext.cacheDir,"runtime-${UUID.randomUUID()}").apply { mkdirs() }
        val prefix="note-chat-runtime-${UUID.randomUUID()}"
        val isolated=object:ContextWrapper(targetContext) { override fun getFilesDir()=directory }
        val vault=Vault(isolated,prefix)
        try {
            vault.openWithoutLock()
            val classes=StateJson.categories(targetContext.assets.open("classes.json").bufferedReader().use { it.readText() })
            val message=Message(text="Private fixture: book a train",category="todos",approved=true)
            val state=ChatState(classes,listOf(message))
            vault.save(state)
            check(vault.load()==state)
            val encrypted=File(directory,"workspace.enc").readBytes()
            check(!encrypted.toString(Charsets.ISO_8859_1).contains(message.text))
            val media=byteArrayOf(1,2,3,4,5,6,7)
            val attachment=vault.importMedia(ByteArrayInputStream(media),"example.bin","application/octet-stream")
            check(vault.readMedia(attachment).use { it.readBytes() }.contentEquals(media))
            vault.close()
            check(!vault.isOpen)
            try { vault.load(); error("Locked data was readable") } catch(_:IllegalStateException) { }
            vault.openWithoutLock(); check(vault.load()==state)
            val damaged=encrypted.copyOf().apply { this[lastIndex]=(this[lastIndex].toInt() xor 1).toByte() }
            File(directory,"workspace.enc").writeBytes(damaged)
            var rejected=false
            try { vault.load() } catch(_:Exception) { rejected=true }
            check(rejected) { "Tampered ciphertext accepted" }
            vault.close()
            output.putString("storage","PASS: Keystore-wrapped key, encrypted snapshot/media, reload, lock, tamper rejection")

            // Use a fresh emulator: never overwrite a real user's workspace.
            if(!File(targetContext.filesDir,"workspace.enc").exists()) {
                val fixture=Vault(targetContext)
                fixture.openWithoutLock()
                val texts=listOf("The city feels quieter after the rain.","What if the library hosted a seed exchange?","Download books for the flight.","Why do leaves change color?","https://example.org/books","const count = 42;","The client approved the roadmap.")
                val seeded=classes.mapIndexed { index,category->Message(text=texts[index],category=category.id,created=System.currentTimeMillis()-(7-index)*60000L,approved=index==0) }
                fixture.save(ChatState(classes,seeded)); fixture.close()
                val activity=startActivitySync(Intent(targetContext,MainActivity::class.java).apply { addFlags(Intent.FLAG_ACTIVITY_NEW_TASK); action=Intent.ACTION_SEND; type="text/plain"; putExtra(Intent.EXTRA_TEXT,"Remember to book the train") })
                val deadline=System.currentTimeMillis()+90000
                var ready=false
                while(System.currentTimeMillis()<deadline && !ready) {
                    runOnMainSync { ready=labels(activity.window.decorView).any { it=="Download books for the flight." } }
                    if(!ready)Thread.sleep(200)
                }
                check(ready) { "Chat did not render the persisted inbox" }
                runOnMainSync {
                    val view=activity.window.decorView
                    val bitmap=Bitmap.createBitmap(view.width,view.height,Bitmap.Config.ARGB_8888)
                    view.draw(Canvas(bitmap))
                    File(targetContext.cacheDir,"runtime-chat.png").outputStream().use { bitmap.compress(Bitmap.CompressFormat.PNG,100,it) }
                    check(labels(view).any { it=="Remember to book the train" }) { "Shared text was not kept in the composer" }
                    activity.finish()
                }
                output.putString("chat","PASS: encrypted workspace opened, inbox rendered, incoming share kept as a draft")
            }
            output.putString("stream","Runtime checks passed\n")
            finish(Activity.RESULT_OK,output)
        } catch(error:Throwable) {
            output.putString("stream",error.stackTraceToString())
            finish(Activity.RESULT_CANCELED,output)
        } finally {
            vault.close(); directory.deleteRecursively()
            val keys=KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
            listOf("$prefix-device","$prefix-biometric").forEach { if(keys.containsAlias(it))keys.deleteEntry(it) }
        }
    }
    private fun labels(view:View):List<String> = when(view) {
        is TextView->listOf(view.text.toString())
        is ViewGroup->(0 until view.childCount).flatMap { labels(view.getChildAt(it)) }
        else->emptyList()
    }
}
