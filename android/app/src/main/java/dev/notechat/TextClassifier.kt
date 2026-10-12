package dev.notechat

import java.security.MessageDigest
import kotlin.math.exp
import kotlin.math.sqrt

interface SentenceEncoder { val identity: String; fun encode(text: String): FloatArray }

class TextClassifier(private val encoder: SentenceEncoder?, private val timing: (Long)->Unit = {}) {
    private val dimensions = 524
    private var categories = emptyList<Category>()
    private var approved = emptyList<Message>()
    private var head: Head? = null
    private var vectors = emptyList<FloatArray>()
    private var splitAssignments = mutableMapOf<String, String>()
    private var starter: Head? = null
    private var starterSignature = ""
    private val cache = object : LinkedHashMap<String, FloatArray>(256, .75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, FloatArray>?) = size > 2048
    }
    private data class Head(val weights: Array<FloatArray>, val bias: FloatArray)
    private val idea = Regex("\\b(what if|we could|idea for|idea:|maybe (we|i|try|build)|how about|could (build|make|add|create))\\b", RegexOption.IGNORE_CASE)
    private val question = Regex("^(who|what|when|where|why|how|which)\\b|^(is|are|can|does|do|should|would|will) (this|that|it|the|a|i|we|you)\\b", RegexOption.IGNORE_CASE)
    private val task = Regex("^(todo\\b|remember to|remind me|(?:i|we) (?:need|have|want) to|buy|send|email|call|book|schedule|finish|fix|pay|submit|write|read|review|update|clean|check|make|prepare|apply|order|download|install|pick up|follow up|learn|watch|take|highlight)\\b", RegexOption.IGNORE_CASE)
    private val code = Regex("```|^\\s*(def \\w+\\(|class \\w+[:(]|(?:const|let|var) \\w+\\s*=|function |Traceback|<[/]?\\w+)|=>", setOf(RegexOption.MULTILINE, RegexOption.IGNORE_CASE))
    private val junk = Regex("^(asdf|qwerty|test test|[.]{3,})$|file attached|image omitted|IMG-\\d+", RegexOption.IGNORE_CASE)

    @Synchronized fun configure(classes: List<Category>, confirmations: List<Message>, saved: PersonalHead? = null, splits: Map<String, String> = emptyMap()) {
        categories = classes
        if(classes.isEmpty())cache.clear()
        approved = confirmations.filter { it.approved && it.deleted == null && it.trainable() && classes.any { c -> c.id == it.category } }
        vectors = approved.map { feature(it.text) }
        if(starterSignature!=signature()) { starter=if(classes.isEmpty())null else fit(seeds()); starterSignature=signature() }
        head=starter
        personalHead = saved != null && saved.signature == signature() && saved.weights.size == classes.size && saved.weights.all { it.size == dimensions } && saved.bias.size == classes.size
        if (personalHead) head = Head(saved!!.weights.map { it.toFloatArray() }.toTypedArray(), saved.bias.toFloatArray())
        splitAssignments = splits.toMutableMap()
    }
    private fun seeds(exclude: Set<String> = emptySet()): List<Pair<String, String>> = categories.flatMap { c ->
        (c.examples.ifEmpty { listOf(c.description) }).filterNot { normalized(it) in exclude }.map { it to c.id }
    }
    private fun feature(text: String): FloatArray = cache.getOrPut(text) {
        val result = FloatArray(dimensions)
        encoder?.encode(text)?.forEachIndexed { index, value -> if (index < 384) result[index] = value * 2 }
        val cues = floatArrayOf(if (code.containsMatchIn(text)) 1f else 0f, if (junk.containsMatchIn(text)) 1f else 0f,
            if (TextLinks.contains(text)) 1f else 0f, if (text.trim().startsWith("- [")) 1f else 0f,
            if (idea.containsMatchIn(text)) 1f else 0f, if (question.containsMatchIn(text) || text.trim().endsWith("?")) 1f else 0f,
            if (task.containsMatchIn(text)) 1f else 0f, if (Regex("tomorrow|today|deadline|due", RegexOption.IGNORE_CASE).containsMatchIn(text)) 1f else 0f,
            (text.length / 500f).coerceAtMost(1f), (text.split(Regex("\\s+")).size / 60f).coerceAtMost(1f),
            if (text.uppercase() == text && text.any { it.isLetter() }) 1f else 0f,
            if (Regex("\\b(think|feel|noticed|realized|wonder)\\b", RegexOption.IGNORE_CASE).containsMatchIn(text)) 1f else 0f)
        cues.copyInto(result, 384)
        val digest = MessageDigest.getInstance("SHA-256")
        Regex("[\\p{L}\\p{N}_]+").findAll(normalized(text).take(6000)).forEach {
            val hash = digest.digest(it.value.toByteArray(Charsets.UTF_8))
            result[396 + (hash[0].toInt() and 127)] += 1f
        }
        val norm = sqrt(result.sliceArray(396 until 524).sumOf { (it * it).toDouble() }).toFloat().coerceAtLeast(1e-12f)
        for (i in 396 until 524) result[i] /= norm
        result
    }
    private fun fit(records: List<Pair<String, String>>): Head {
        val rows = records.map { feature(it.first) to categories.indexOfFirst { c -> c.id == it.second } }.filter { it.second >= 0 }
        val weights = Array(categories.size) { FloatArray(dimensions) }
        val bias = FloatArray(categories.size)
        if (rows.isEmpty()) return Head(weights, bias)
        val counts = IntArray(categories.size)
        rows.forEach { counts[it.second]++ }
        repeat(180) { step ->
            val gradient = Array(categories.size) { FloatArray(dimensions) }
            val biases = FloatArray(categories.size)
            for ((x, actual) in rows) {
                val probabilities = probabilities(x, Head(weights, bias))
                val balance = rows.size.toFloat() / (categories.size * counts[actual].coerceAtLeast(1))
                for (c in categories.indices) {
                    val error = (probabilities[c] - if (actual == c) 1f else 0f) * balance
                    biases[c] += error
                    for (d in x.indices) gradient[c][d] += error * x[d]
                }
            }
            val rate = .3f / (1 + step / 200f)
            for (c in categories.indices) {
                bias[c] -= rate * biases[c] / rows.size
                for (d in 0 until dimensions) weights[c][d] -= rate * (gradient[c][d] / rows.size + .003f * weights[c][d])
            }
        }
        return Head(weights, bias)
    }
    private fun probabilities(vector: FloatArray, head: Head): FloatArray {
        val logits = FloatArray(head.bias.size) { c -> head.bias[c] + vector.indices.sumOf { (vector[it] * head.weights[c][it]).toDouble() }.toFloat() }
        val maximum = logits.maxOrNull() ?: 0f
        val exp = logits.map { exp((it - maximum).toDouble()).toFloat() }
        val sum = exp.sum().coerceAtLeast(1e-12f)
        return exp.map { it / sum }.toFloatArray()
    }
    private fun predict(message: Message, fitted: Head?, memory: List<Message>, remembered: List<FloatArray>, personal: Boolean): Suggestion {
        if (!message.trainable()) return Suggestion(null, "Choose a category for this attachment.")
        val allowed = categories.filterNot { it.id in message.rejected }
        if (allowed.isEmpty()) return Suggestion(null, "No suggestions left. Choose the category yourself.")
        val exact = memory.lastOrNull { normalized(it.text) == normalized(message.text) && it.category in allowed.map { c -> c.id } }
        if (exact != null) return Suggestion(exact.category, "Matches a message you approved.")
        val x = feature(message.text)
        if (encoder != null) {
            val nearest = memory.indices.filter { memory[it].category in allowed.map { c -> c.id } }.maxByOrNull { index ->
                (0 until 384).sumOf { (x[it] * remembered[index][it] / 4).toDouble() }
            }
            if (nearest != null && (0 until 384).sumOf { (x[it] * remembered[nearest][it] / 4).toDouble() } >= .86)
                return Suggestion(memory[nearest].category, "Similar to a message you approved.")
        }
        fun cue(id: String, reason: String) = if (allowed.any { it.id == id }) Suggestion(id, reason) else null
        if (!personal) {
            if (code.containsMatchIn(message.text) || junk.containsMatchIn(message.text)) cue("junk", "Code or placeholder text.")?.let { return it }
            if (message.text.trim().startsWith("- [")) cue("todos", "Contains a task checkbox.")?.let { return it }
            if (idea.containsMatchIn(message.text)) cue("ideas", "Proposes a possibility or change.")?.let { return it }
            if (question.containsMatchIn(message.text) || message.text.trim().endsWith("?")) cue("questions", "Question wording.")?.let { return it }
            if (task.containsMatchIn(message.text)) cue("todos", "An action to complete.")?.let { return it }
            if (TextLinks.contains(message.text)) cue("links", "A saved web reference.")?.let { return it }
        }
        if (fitted == null) return Suggestion(allowed.first().id, "Choose or approve a category.")
        val p = probabilities(x, fitted)
        val winner = categories.indices.filter { categories[it].id !in message.rejected }.maxBy { p[it] }
        return Suggestion(categories[winner].id, "The small classifier favors ${categories[winner].title.lowercase()}.")
    }
    @Synchronized fun train(): TrainingReport {
        require(categories.isNotEmpty()) { "Add a category first" }
        val groups = approved.distinctBy { normalized(it.text) }.groupBy { it.category }
        require(categories.all { (groups[it.id]?.size ?: 0) >= 6 }) { "Approve at least six different text messages in each category." }
        categories.forEach { c ->
            val rows = groups.getValue(c.id)
            val unknown = rows.filter { normalized(it.text) !in splitAssignments }.sortedBy { stableHash(it.text) }
            val existingTests = rows.count { splitAssignments[normalized(it.text)] == "test" }
            val required = ((rows.size / 5).coerceAtLeast(1) - existingTests).coerceAtLeast(0)
            unknown.forEachIndexed { i, row -> splitAssignments[normalized(row.text)] = if (i < required) "test" else "train" }
        }
        val test = approved.filter { splitAssignments[normalized(it.text)] == "test" }
        val held = test.map { normalized(it.text) }.toSet()
        val training = approved.filterNot { normalized(it.text) in held }
        require(categories.all { c -> test.any { it.category == c.id } && training.any { it.category == c.id } }) { "Approve more new messages to keep training and test examples in every category." }
        val seeds = seeds(held)
        val baseline = fit(seeds)
        val candidate = fit(seeds + training.map { it.text to it.category!! })
        val memory = training.map { feature(it.text) }
        fun score(model: Head, personal: Boolean): Double {
            val predictions = test.map { predict(it.copy(rejected = emptySet()), model, training, memory, personal).category }
            return categories.indices.sumOf { i ->
                val id = categories[i].id
                val tp = test.indices.count { test[it].category == id && predictions[it] == id }.toDouble()
                val precision = tp / predictions.count { it == id }.coerceAtLeast(1)
                val recall = tp / test.count { it.category == id }.coerceAtLeast(1)
                if (precision + recall == 0.0) 0.0 else 2 * precision * recall / (precision + recall)
            } / categories.size
        }
        val baseScore = score(baseline, false)
        val candidateScore = score(candidate, true)
        val adopted = candidateScore >= baseScore
        head = if (adopted) candidate else starter
        personalHead = adopted
        return TrainingReport(approved.size, test.size, baseScore, candidateScore, adopted)
    }
    private var personalHead = false
    fun learned() = if (personalHead) head?.let { PersonalHead(signature(), it.weights.map { row -> row.toList() }, it.bias.toList()) } else null
    fun splits(): Map<String, String> = splitAssignments.toMap()
    private fun signature() = signature(categories,encoder?.identity)
    companion object {
        private fun signature(classes: List<Category>, identity: String?) = MessageDigest.getInstance("SHA-256").digest(normalized((identity ?: "lexical")+classes.joinToString("|") { "${it.id}:${it.description}:${it.examples.joinToString()}" }).toByteArray()).joinToString("") { "%02x".format(it) }
        fun compatible(saved: PersonalHead, classes: List<Category>, identity: String?) = saved.signature==signature(classes,identity) && saved.weights.size==classes.size && saved.weights.all { row->row.size==524 && row.all(Float::isFinite) } && saved.bias.size==classes.size && saved.bias.all(Float::isFinite)
    }
    @Synchronized fun classify(message: Message): Suggestion {
        val start=System.nanoTime()
        return try { predict(message,head,approved,vectors,personalHead) } finally { timing(System.nanoTime()-start) }
    }
    private fun stableHash(text: String) = MessageDigest.getInstance("SHA-256").digest(normalized(text).toByteArray()).joinToString("") { "%02x".format(it) }
}
