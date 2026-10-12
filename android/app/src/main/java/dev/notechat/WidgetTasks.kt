package dev.notechat

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.AtomicFile
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/** An opt-in projection for the launcher, encrypted with a separate device key. */
class WidgetTasks(context: Context) {
    private val file=AtomicFile(File(context.filesDir,"widget-tasks.enc"))
    private fun key(): SecretKey {
        val store=KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        val alias="note-chat-widget"
        (store.getKey(alias,null) as? SecretKey)?.let { return it }
        return KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES,"AndroidKeyStore").apply {
            init(KeyGenParameterSpec.Builder(alias,KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM).setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE).setKeySize(256).build())
        }.generateKey()
    }
    fun publish(state: ChatState) = synchronized(guard) {
        if(!state.preferences.widgetWhenLocked) { file.delete(); return@synchronized }
        val rows=JSONArray(project(state).map { JSONObject().put("id",it.id).put("text",it.text).put("done",it.done).put("completedAt",it.completedAt ?: JSONObject.NULL).put("starred",it.starred).put("starArchiveAt",it.starArchiveAt ?: JSONObject.NULL) })
        val cipher=Cipher.getInstance("AES/GCM/NoPadding").apply { init(Cipher.ENCRYPT_MODE,key()) }
        val data=cipher.iv+cipher.doFinal(rows.toString().toByteArray(Charsets.UTF_8))
        val output=file.startWrite()
        try { output.write(data); file.finishWrite(output) } catch(error: Exception) { file.failWrite(output); throw error }
    }
    fun read(): List<Message>? = synchronized(guard) {
        if(!file.baseFile.exists())return@synchronized null
        val data=file.readFully()
        val cipher=Cipher.getInstance("AES/GCM/NoPadding").apply { init(Cipher.DECRYPT_MODE,key(),GCMParameterSpec(128,data.copyOfRange(0,12))) }
        val rows=JSONArray(cipher.doFinal(data.copyOfRange(12,data.size)).toString(Charsets.UTF_8))
        (0 until rows.length()).map { i -> rows.getJSONObject(i).let { Message(id=it.getString("id"),text=it.getString("text"),category="todos",done=it.getBoolean("done"),completedAt=if(it.isNull("completedAt"))null else it.getLong("completedAt"),starred=it.optBoolean("starred"),starArchiveAt=if(it.isNull("starArchiveAt"))null else it.getLong("starArchiveAt")) } }.filter { it.visible() }
    }
    companion object {
        private val guard=Any()
        fun project(state: ChatState, now: Long=System.currentTimeMillis()) = state.messages.filter { it.visible(now) && it.category=="todos" }
            .map { Message(id=it.id,text=it.text.ifBlank { it.attachment?.name.orEmpty() },category="todos",done=it.done,completedAt=it.completedAt,starred=it.starred,starArchiveAt=it.starArchiveAt) }
    }
}
