package dev.notechat

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.AtomicFile
import java.io.File
import java.io.InputStream
import java.security.KeyStore
import java.security.SecureRandom
import javax.crypto.Cipher
import javax.crypto.CipherInputStream
import javax.crypto.CipherOutputStream
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.SecretKeySpec

class Vault(private val context: Context, private val keyPrefix: String = "note-chat") {
    companion object { private val diskGuard=Any() }
    private val wrapped = AtomicFile(File(context.filesDir, "workspace.key"))
    private val snapshot = AtomicFile(File(context.filesDir, "workspace.enc"))
    private val media = File(context.filesDir, "media").apply { mkdirs() }
    private val keys = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
    private var dataKey: ByteArray? = null
    val lockedStorage: Boolean get() = synchronized(diskGuard) { wrapped.baseFile.exists() && wrapped.readFully()[0].toInt() == 1 }
    val isOpen get() = dataKey != null

    private fun key(authenticated: Boolean): SecretKey {
        val alias = if (authenticated) "$keyPrefix-biometric" else "$keyPrefix-device"
        (keys.getKey(alias, null) as? SecretKey)?.let { return it }
        val generator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore")
        val builder = KeyGenParameterSpec.Builder(alias, KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
            .setBlockModes(KeyProperties.BLOCK_MODE_GCM).setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE).setKeySize(256)
        if (authenticated) builder.setUserAuthenticationRequired(true).setUserAuthenticationParameters(0,
            KeyProperties.AUTH_BIOMETRIC_STRONG or KeyProperties.AUTH_DEVICE_CREDENTIAL).setInvalidatedByBiometricEnrollment(false)
        generator.init(builder.build())
        return generator.generateKey()
    }
    @Synchronized fun openWithoutLock(): Unit = synchronized(diskGuard) {
        require(!lockedStorage)
        if (wrapped.baseFile.exists()) {
            val data = readWrapped()
            dataKey = wrappingCipher(false, false, data.copyOfRange(1, 13)).doFinal(data.copyOfRange(13, data.size))
        } else {
            dataKey = ByteArray(32).also { SecureRandom().nextBytes(it) }
            finishWrapping(wrappingCipher(false, true), false)
        }
    }
    fun unlockCipher(): Cipher {
        val data = readWrapped()
        return wrappingCipher(true, false, data.copyOfRange(1, 13))
    }
    @Synchronized fun finishUnlock(cipher: Cipher) {
        val data = readWrapped()
        dataKey = cipher.doFinal(data.copyOfRange(13, data.size))
    }
    fun enableLockCipher() = wrappingCipher(true, true)
    @Synchronized fun finishWrapping(cipher: Cipher, locked: Boolean) {
        val encrypted = cipher.doFinal(dataKey ?: error("Unlock first"))
        atomic(wrapped, byteArrayOf(if (locked) 1 else 0) + cipher.iv + encrypted)
    }
    @Synchronized fun disableLock() = finishWrapping(wrappingCipher(false, true), false)
    private fun wrappingCipher(auth: Boolean, encrypt: Boolean, iv: ByteArray? = null) = Cipher.getInstance("AES/GCM/NoPadding").apply {
        if (encrypt) init(Cipher.ENCRYPT_MODE, key(auth)) else init(Cipher.DECRYPT_MODE, key(auth), GCMParameterSpec(128, iv))
    }
    private fun cipher(encrypt: Boolean, iv: ByteArray? = null) = Cipher.getInstance("AES/GCM/NoPadding").apply {
        val key = SecretKeySpec(dataKey ?: error("Workspace locked"), "AES")
        if (encrypt) init(Cipher.ENCRYPT_MODE, key) else init(Cipher.DECRYPT_MODE, key, GCMParameterSpec(128, iv))
    }
    @Synchronized fun load(): ChatState {
        if (!snapshot.baseFile.exists()) return ChatState(StateJson.categories(context.assets.open("classes.json").bufferedReader().use { it.readText() }))
        val encrypted = synchronized(diskGuard) { snapshot.readFully() }
        return StateJson.decode(cipher(false, encrypted.copyOfRange(0, 12)).doFinal(encrypted.copyOfRange(12, encrypted.size)))
    }
    @Synchronized fun save(state: ChatState) {
        val cipher = cipher(true)
        atomic(snapshot, cipher.iv + cipher.doFinal(StateJson.encode(state)))
    }
    @Synchronized fun importMedia(input: InputStream, name: String, mime: String, duration: Long = 0): Attachment {
        val id = java.util.UUID.randomUUID().toString()
        val file = File(media, id)
        val cipher = cipher(true)
        var count = 0L
        try {
            file.outputStream().use { raw ->
                raw.write(cipher.iv)
                CipherOutputStream(raw, cipher).use { encrypted ->
                    val buffer = ByteArray(64 * 1024)
                    while (true) {
                        val size = input.read(buffer)
                        if (size < 0) break
                        count += size
                        require(count <= 50L * 1024 * 1024) { "Attachments are limited to 50 MB" }
                        encrypted.write(buffer, 0, size)
                    }
                }
            }
        } catch (error: Exception) { file.delete(); throw error }
        return Attachment(id, name.substringAfterLast('/').substringAfterLast('\\').take(120).ifBlank { "attachment" }, mime, count, duration)
    }
    @Synchronized fun readMedia(attachment: Attachment): InputStream {
        require(attachment.file.matches(Regex("[a-f0-9-]{36}")))
        val raw = File(media, attachment.file).inputStream()
        try {
            val iv = ByteArray(12)
            require(raw.read(iv) == 12)
            return CipherInputStream(raw, cipher(false, iv))
        } catch (error: Exception) { raw.close(); throw error }
    }
    fun removeMedia(attachment: Attachment) { if (attachment.file.matches(Regex("[a-f0-9-]{36}"))) File(media, attachment.file).delete() }
    @Synchronized fun close() { dataKey?.fill(0); dataKey = null }
    private fun readWrapped() = synchronized(diskGuard) { wrapped.readFully() }
    private fun atomic(file: AtomicFile, bytes: ByteArray): Unit = synchronized(diskGuard) {
        val output = file.startWrite()
        try { output.write(bytes); file.finishWrite(output) }
        catch (error: Exception) { file.failWrite(output); throw error }
    }
}
