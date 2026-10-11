package dev.notechat

import org.json.JSONArray
import org.json.JSONObject

object StateJson {
    private fun JSONArray.strings() = (0 until length()).map { getString(it) }
    private fun JSONObject.optional(key: String) = if (isNull(key)) null else optString(key).takeIf { it.isNotEmpty() }
    fun categories(json: String): List<Category> {
        val array = JSONArray(json)
        return (0 until array.length()).map { i -> array.getJSONObject(i).let { c -> Category(
            c.optString("id", c.optString("name")), c.getString("title"), c.getString("description"),
            c.optJSONArray("examples")?.strings() ?: emptyList(), c.optBoolean("pinned")) } }
    }
    fun encode(state: ChatState): ByteArray {
        val classes = JSONArray(state.categories.map { c -> JSONObject().put("id", c.id).put("title", c.title)
            .put("description", c.description).put("examples", JSONArray(c.examples)).put("pinned", c.pinned) })
        val messages = JSONArray(state.messages.map { m -> JSONObject().put("id", m.id).put("text", m.text)
            .put("created", m.created).put("category", m.category ?: JSONObject.NULL).put("approved", m.approved)
            .put("rejected", JSONArray(m.rejected.toList())).put("replyTo", m.replyTo ?: JSONObject.NULL)
            .put("starred", m.starred).put("done", m.done).put("edited", m.edited).put("deleted", m.deleted ?: JSONObject.NULL)
            .put("completedAt",m.completedAt ?: JSONObject.NULL).put("starArchiveAt",m.starArchiveAt ?: JSONObject.NULL).put("transcribed",m.transcribed)
            .put("reason", m.reason).put("manual", m.manual).put("sender",m.sender ?: JSONObject.NULL).put("attachment", m.attachment?.let { a -> JSONObject()
                .put("file", a.file).put("name", a.name).put("mime", a.mime).put("bytes", a.bytes).put("duration", a.duration).put("waveform",JSONArray(a.waveform)) } ?: JSONObject.NULL) })
        val report = state.report?.let { r -> JSONObject().put("confirmed", r.confirmed).put("heldOut", r.heldOut)
            .put("baseline", r.baseline).put("candidate", r.candidate).put("adopted", r.adopted).put("timestamp", r.timestamp) }
        val model = state.model?.let { m -> JSONObject().put("signature", m.signature)
            .put("weights", JSONArray(m.weights.map { JSONArray(it) })).put("bias", JSONArray(m.bias)) }
        return JSONObject().put("version", 1).put("categories", classes).put("messages", messages)
            .put("report", report ?: JSONObject.NULL).put("model", model ?: JSONObject.NULL).put("splits", JSONObject(state.splits))
            .put("preferences",JSONObject().put("preserveImportDates",state.preferences.preserveImportDates).put("appearance",state.preferences.appearance))
            .put("navigation",JSONObject().put("route",state.navigation.route).put("anchor",state.navigation.anchor ?: JSONObject.NULL).put("index",state.navigation.index).put("offset",state.navigation.offset).put("scroll",state.navigation.scroll))
            .put("draft",state.draft).put("replyDraft",state.replyDraft ?: JSONObject.NULL).toString().toByteArray(Charsets.UTF_8)
    }
    fun decode(bytes: ByteArray): ChatState {
        val root = JSONObject(bytes.toString(Charsets.UTF_8))
        require(root.getInt("version") == 1) { "Unsupported workspace version" }
        val array = root.getJSONArray("messages")
        val messages = (0 until array.length()).map { i -> array.getJSONObject(i).let { m ->
            val attachment = m.optJSONObject("attachment")?.let { a -> Attachment(a.getString("file"), a.getString("name"), a.getString("mime"), a.getLong("bytes"), a.optLong("duration"),a.optJSONArray("waveform")?.let { wave -> (0 until wave.length()).map { wave.getInt(it) } } ?: emptyList()) }
            Message(m.getString("id"), m.getString("text"), m.getLong("created"), m.optional("category"), m.optBoolean("approved"),
                m.optJSONArray("rejected")?.strings()?.toSet() ?: emptySet(), attachment, m.optional("replyTo"), m.optBoolean("starred"),
                m.optBoolean("done"), m.optBoolean("edited"), if (m.isNull("deleted")) null else m.getLong("deleted"), m.optString("reason"), m.optBoolean("manual"),m.optional("sender"),if(m.isNull("completedAt"))null else m.getLong("completedAt"),
                if(m.isNull("starArchiveAt"))null else m.getLong("starArchiveAt"),m.optBoolean("transcribed"))
        } }
        val report = root.optJSONObject("report")?.let { r -> TrainingReport(r.getInt("confirmed"), r.getInt("heldOut"), r.getDouble("baseline"), r.getDouble("candidate"), r.getBoolean("adopted"), r.getLong("timestamp")) }
        val model = root.optJSONObject("model")?.let { m ->
            val weights = m.getJSONArray("weights")
            val bias = m.getJSONArray("bias")
            PersonalHead(m.getString("signature"), (0 until weights.length()).map { i -> weights.getJSONArray(i).let { row -> (0 until row.length()).map { row.getDouble(it).toFloat() } } },
                (0 until bias.length()).map { bias.getDouble(it).toFloat() })
        }
        val split = root.optJSONObject("splits")
        val assignments = split?.keys()?.asSequence()?.associateWith { split.getString(it) } ?: emptyMap()
        val navigation=root.optJSONObject("navigation")?.let { Navigation(it.optString("route","home"),it.optional("anchor"),it.optInt("index").coerceAtLeast(0),it.optInt("offset"),it.optInt("scroll").coerceAtLeast(0)) } ?: Navigation()
        return ChatState(categories(root.getJSONArray("categories").toString()), messages, report, model, assignments,root.optString("draft"),root.optional("replyDraft"),root.optJSONObject("preferences")?.let { Preferences(it.optBoolean("preserveImportDates"),it.optString("appearance","system")) } ?: Preferences(),navigation)
    }
}
