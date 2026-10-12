package dev.notechat

import org.json.JSONObject
import java.io.File
import java.io.FilterOutputStream
import java.io.InputStream
import java.io.OutputStream
import java.security.MessageDigest
import java.security.SecureRandom
import java.util.Base64
import java.util.UUID
import java.util.zip.GZIPInputStream
import java.util.zip.GZIPOutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipFile
import java.util.zip.ZipOutputStream
import javax.crypto.Cipher
import javax.crypto.CipherOutputStream
import javax.crypto.SecretKeyFactory
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.PBEKeySpec
import javax.crypto.spec.SecretKeySpec

object BackupArchive {
    data class Restored(val state: ChatState, val kind: String, val fingerprint: String, val media: Map<String,File>, val staging: File) {
        fun close() { staging.deleteRecursively() }
    }
    private const val iterations=600000
    private const val maxWorkspace=16*1024*1024
    private const val maxMedia=50L*1024*1024
    private const val maxTotal=512L*1024*1024
    fun attachments(state: ChatState) = (state.messages.mapNotNull { it.attachment }+state.categories.mapNotNull { it.photo }+state.chats.mapNotNull { it.photo }).distinctBy { it.file }
    fun portable(state: ChatState) = state.copy(markdown=MarkdownState(),navigation=Navigation(),draft="",replyDraft=null,editDraftId=null,preEditDraft="",drafts=emptyMap(),locations=emptyMap(),guide=GuideProgress(),homeTab="home")
    fun validate(state: ChatState) {
        require(state.categories.size<=32 && state.chats.size<=256 && state.messages.size<=50000)
        require(state.categories.map { it.id }.distinct().size==state.categories.size && state.chats.map { it.id }.distinct().size==state.chats.size)
        require(state.chats.none { chat->state.categories.any { it.id==chat.id } })
        require(state.messages.map { it.id }.distinct().size==state.messages.size) { "Backup contains duplicate message IDs" }
        require(state.categories.all { it.id.matches(Regex("[A-Za-z0-9-]{1,64}")) && it.title.isNotBlank() && it.title.length<=60 && it.description.length<=2000 && it.examples.size<=30 && it.examples.all { text -> text.length<=2000 } })
        require(state.chats.all { it.id.matches(Regex("[A-Za-z0-9-]{1,64}")) && it.title.isNotBlank() && it.title.length<=60 })
        require(state.messages.all { it.id.matches(Regex("[A-Za-z0-9-]{1,64}")) && it.text.length<=20000 && (it.category==null || state.categories.any { c->c.id==it.category }) && (it.chatId==null || state.chats.any { c->c.id==it.chatId }) })
        require(attachments(state).all { it.file.matches(Regex("[a-f0-9-]{36}")) && it.bytes in 0..maxMedia && it.name.length<=120 })
        state.model?.let { require(it.weights.size==state.categories.size && it.bias.size==state.categories.size && it.weights.all { row->row.size==524 && row.all(Float::isFinite) } && it.bias.all(Float::isFinite)) { "Invalid classifier weights" } }
    }
    private fun key(password: CharArray, salt: ByteArray): SecretKeySpec {
        require(password.size>=8) { "Use a backup password with at least eight characters" }
        val spec=PBEKeySpec(password,salt,iterations,256)
        return try { SecretKeySpec(SecretKeyFactory.getInstance("PBKDF2WithHmacSHA256").generateSecret(spec).encoded,"AES") } finally { spec.clearPassword() }
    }
    fun write(file: File, state: ChatState, password: CharArray, kind: String="workspace", media: (Attachment)->InputStream) {
        require(kind in listOf("workspace","model")); validate(state)
        if(kind=="model")require(state.messages.isEmpty() && attachments(state).isEmpty() && state.model!=null && state.quiz==null && state.splits.isEmpty()) { "Model export must exclude personal examples and media" }
        val snapshot=portable(state)
        val salt=ByteArray(16).also { SecureRandom().nextBytes(it) }; val key=key(password,salt)
        try { ZipOutputStream(file.outputStream()).use { zip ->
            zip.putNextEntry(ZipEntry("manifest.json"))
            zip.write(JSONObject().put("format","note-chat-backup").put("version",1).put("kind",kind).put("iterations",iterations)
                .put("salt",Base64.getEncoder().encodeToString(salt)).toString().toByteArray())
            zip.closeEntry()
            fun encrypted(name: String, write: (OutputStream)->Unit) {
                zip.putNextEntry(ZipEntry(name))
                val iv=ByteArray(12).also { SecureRandom().nextBytes(it) }; zip.write(iv)
                val cipher=Cipher.getInstance("AES/GCM/NoPadding").apply { init(Cipher.ENCRYPT_MODE,key,GCMParameterSpec(128,iv)); updateAAD(name.toByteArray()) }
                CipherOutputStream(object:FilterOutputStream(zip) { override fun close() { flush() } },cipher).use(write)
                zip.closeEntry()
            }
            encrypted("workspace.enc") { output -> GZIPOutputStream(output).use { it.write(StateJson.encode(snapshot)) } }
            attachments(snapshot).forEach { attachment -> encrypted("media/${attachment.file}.enc") { out -> media(attachment).use { it.copyTo(out) } } }
        } } catch(error: Exception) { file.delete(); throw error }
    }
    fun kind(file: File): String? = ZipFile(file).use { zip -> zip.getEntry("manifest.json")?.let { entry ->
        val json=JSONObject(zip.getInputStream(entry).use { it.limited(65536).toString(Charsets.UTF_8) })
        if(json.optString("format")=="note-chat-backup")json.getString("kind") else null
    } }
    fun read(file: File, password: CharArray, cache: File): Restored {
        require(file.length()<=maxTotal) { "Backup is larger than 512 MB" }
        val staging=File(cache,"restore-${UUID.randomUUID()}").apply { mkdirs() }
        try { ZipFile(file).use { zip ->
            val entries=zip.entries().asSequence().toList()
            require(entries.size<=5000 && entries.map { it.name }.distinct().size==entries.size)
            require(entries.all { it.name in listOf("manifest.json","workspace.enc") || it.name.matches(Regex("media/[a-f0-9-]{36}\\.enc")) }) { "Unexpected backup entry" }
            val manifest=JSONObject(zip.getInputStream(zip.getEntry("manifest.json") ?: error("Backup manifest missing")).use { it.limited(65536).toString(Charsets.UTF_8) })
            require(manifest.getString("format")=="note-chat-backup" && manifest.getInt("version")==1 && manifest.getInt("iterations")==iterations)
            val kind=manifest.getString("kind"); require(kind in listOf("workspace","model"))
            val salt=Base64.getDecoder().decode(manifest.getString("salt")); require(salt.size==16)
            val key=key(password,salt)
            fun decryptTo(name: String, target: File, limit: Long): Long {
                var written=0L
                zip.getInputStream(zip.getEntry(name) ?: error("Missing backup entry: $name")).use { input ->
                    val iv=ByteArray(12); java.io.DataInputStream(input).readFully(iv)
                    val cipher=Cipher.getInstance("AES/GCM/NoPadding").apply { init(Cipher.DECRYPT_MODE,key,GCMParameterSpec(128,iv)); updateAAD(name.toByteArray()) }
                    target.outputStream().use { output ->
                        fun append(bytes: ByteArray?) { if(bytes!=null) { written+=bytes.size; require(written<=limit) { "Decrypted entry is too large" }; output.write(bytes) } }
                        val buffer=ByteArray(8192); var consumed=0L
                        while(true) { val n=input.read(buffer); if(n<0)break; consumed+=n; require(consumed<=limit+16); append(cipher.update(buffer,0,n)) }
                        // Explicit finalization propagates GCM authentication failures on Android.
                        append(cipher.doFinal())
                    }
                }
                return written
            }
            val workspace=File(staging,"workspace.gz")
            decryptTo("workspace.enc",workspace,maxWorkspace.toLong()+65536)
            val state=workspace.inputStream().use { input -> GZIPInputStream(input).use { StateJson.decode(it.limited(maxWorkspace)) } }
            workspace.delete()
            validate(state)
            if(kind=="model")require(state.messages.isEmpty() && state.model!=null && attachments(state).isEmpty())
            var total=0L
            val files=attachments(state).associate { attachment ->
                val target=File(staging,attachment.file)
                val count=decryptTo("media/${attachment.file}.enc",target,maxMedia)
                total+=count; require(total<=maxTotal && count==attachment.bytes) { "Attachment size differs from backup metadata" }
                attachment.file to target
            }
            val digest=MessageDigest.getInstance("SHA-256"); file.inputStream().use { input -> val buffer=ByteArray(8192); while(true) { val n=input.read(buffer); if(n<0)break; digest.update(buffer,0,n) } }
            return Restored(portable(state),kind,digest.digest().joinToString("") { "%02x".format(it) },files,staging)
        } } catch(error: Exception) { staging.deleteRecursively(); throw error }
    }
    fun merge(current: ChatState, restored: Restored): ChatState {
        if(current.messages.isEmpty())return restored.state.copy(importedArchives=current.importedArchives+restored.state.importedArchives+restored.fingerprint)
        val known=current.messages.map { it.id }.toSet()
        val categories=current.categories+restored.state.categories.filter { c -> current.categories.none { it.id==c.id } }
        val chats=current.chats+restored.state.chats.filter { c -> current.chats.none { it.id==c.id } }
        require(categories.size<=32 && chats.size<=256)
        return current.copy(categories=categories,chats=chats,messages=current.messages+restored.state.messages.filterNot { it.id in known },
            model=if(categories==current.categories)current.model else null,report=if(categories==current.categories)current.report else null,
            importedArchives=current.importedArchives+restored.fingerprint)
    }
    private fun InputStream.limited(max: Int): ByteArray {
        val output=java.io.ByteArrayOutputStream(); val buffer=ByteArray(8192)
        while(true) { val size=read(buffer); if(size<0)break; require(output.size()+size<=max) { "Backup entry is too large" }; output.write(buffer,0,size) }
        return output.toByteArray()
    }
}
