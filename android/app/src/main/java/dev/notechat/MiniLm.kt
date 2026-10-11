package dev.notechat

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.net.URL
import java.nio.LongBuffer
import java.security.MessageDigest
import kotlin.math.sqrt

class MiniLm(context: Context) : SentenceEncoder, AutoCloseable {
    override val identity = CHECKSUM
    private val environment = OrtEnvironment.getEnvironment().apply { setTelemetry(false) }
    private val session = OrtSession.SessionOptions().use { options ->
        options.setIntraOpNumThreads(2)
        options.setInterOpNumThreads(1)
        environment.createSession(modelFile(context).absolutePath, options)
    }
    private val tokenizer = WordPiece(context.assets.open("vocab.txt").bufferedReader().use { reader -> reader.readLines().mapIndexed { i, word -> word to i }.toMap() })
    @Synchronized override fun encode(text: String): FloatArray {
        val tokens = tokenizer.tokens(text)
        val chunks = tokens.toList().chunked(254).ifEmpty { listOf(emptyList()) }
        val vector = FloatArray(384)
        for (chunk in chunks) {
            val ids = (listOf(101L) + chunk + 102L).toLongArray()
            val shape = longArrayOf(1, ids.size.toLong())
            val inputs = mapOf("input_ids" to ids, "attention_mask" to LongArray(ids.size) { 1 }, "token_type_ids" to LongArray(ids.size))
                .filterKeys { it in session.inputNames }.mapValues { OnnxTensor.createTensor(environment, LongBuffer.wrap(it.value), shape) }
            try {
                session.run(inputs).use { output ->
                    @Suppress("UNCHECKED_CAST") val hidden = output[0].value as Array<Array<FloatArray>>
                    hidden[0].forEach { token -> token.forEachIndexed { i, value -> vector[i] += value } }
                }
            } finally { inputs.values.forEach { it.close() } }
        }
        val norm = sqrt(vector.sumOf { (it * it).toDouble() }).toFloat().coerceAtLeast(1e-12f)
        for (i in vector.indices) vector[i] /= norm
        return vector
    }
    override fun close() = session.close()
    companion object {
        const val CHECKSUM = "4278337fd0ff3c68bfb6291042cad8ab363e1d9fbc43dcb499fe91c871902474"
        private const val MODEL_URL = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/onnx/model_qint8_arm64.onnx?download=true"
        fun modelFile(context: Context) = File(context.filesDir, "minilm.onnx")
        fun download(context: Context, progress: (Int) -> Unit) {
            val target = modelFile(context)
            val part = File(target.parentFile, "minilm.part")
            val connection = URL(MODEL_URL).openConnection().apply { connectTimeout = 30000; readTimeout = 30000 }
            val digest = MessageDigest.getInstance("SHA-256")
            try {
                connection.getInputStream().use { input -> part.outputStream().use { output ->
                    val buffer = ByteArray(64 * 1024)
                    var total = 0L
                    while (true) {
                        val count = input.read(buffer)
                        if (count < 0) break
                        total += count
                        require(total <= 40L * 1024 * 1024) { "Unexpected model size" }
                        digest.update(buffer, 0, count)
                        output.write(buffer, 0, count)
                        progress((total * 100 / connection.contentLengthLong.coerceAtLeast(1)).toInt().coerceIn(0, 100))
                    }
                } }
                require(digest.digest().joinToString("") { "%02x".format(it) } == CHECKSUM) { "Model checksum failed" }
                check(part.renameTo(target)) { "Could not save model" }
            } finally { part.delete() }
        }
    }
}
