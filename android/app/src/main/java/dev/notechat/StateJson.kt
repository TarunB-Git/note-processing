package dev.notechat

import org.json.JSONArray
import org.json.JSONObject

object StateJson {
    private fun JSONArray.strings() = (0 until length()).map { getString(it) }
    private fun JSONObject.optional(key: String) = if (isNull(key)) null else optString(key).takeIf { it.isNotEmpty() }
    private fun attachment(a: JSONObject) = Attachment(a.getString("file"),a.getString("name"),a.getString("mime"),a.getLong("bytes"),a.optLong("duration"),a.optJSONArray("waveform")?.let { wave -> (0 until wave.length()).map { wave.getInt(it) } } ?: emptyList())
    private fun attachmentJson(a: Attachment) = JSONObject().put("file",a.file).put("name",a.name).put("mime",a.mime).put("bytes",a.bytes).put("duration",a.duration).put("waveform",JSONArray(a.waveform))
    private fun navigation(value: Navigation) = JSONObject().put("route",value.route).put("anchor",value.anchor ?: JSONObject.NULL).put("index",value.index).put("offset",value.offset).put("scroll",value.scroll)
    private fun navigation(value: JSONObject) = Navigation(value.optString("route","home"),value.optional("anchor"),value.optInt("index").coerceAtLeast(0),value.optInt("offset"),value.optInt("scroll").coerceAtLeast(0))
    fun categories(json: String): List<Category> {
        val array = JSONArray(json)
        return (0 until array.length()).map { i -> array.getJSONObject(i).let { c -> Category(
            c.optString("id", c.optString("name")), c.getString("title"), c.getString("description"),
            c.optJSONArray("examples")?.strings() ?: emptyList(), c.optBoolean("pinned"), c.optional("icon"), c.optJSONObject("photo")?.let(::attachment)) } }
    }
    fun encode(state: ChatState): ByteArray {
        val classes = JSONArray(state.categories.map { c -> JSONObject().put("id", c.id).put("title", c.title)
            .put("description", c.description).put("examples", JSONArray(c.examples)).put("pinned", c.pinned).put("icon",c.icon ?: JSONObject.NULL).put("photo",c.photo?.let(::attachmentJson) ?: JSONObject.NULL) })
        val messages = JSONArray(state.messages.map { m -> JSONObject().put("id", m.id).put("text", m.text)
            .put("created", m.created).put("chatId",m.chatId ?: JSONObject.NULL).put("remindAt",m.remindAt ?: JSONObject.NULL).put("translation",m.translation ?: JSONObject.NULL).put("extracted",m.extracted).put("category", m.category ?: JSONObject.NULL).put("approved", m.approved)
            .put("rejected", JSONArray(m.rejected.toList())).put("replyTo", m.replyTo ?: JSONObject.NULL)
            .put("starred", m.starred).put("done", m.done).put("edited", m.edited).put("deleted", m.deleted ?: JSONObject.NULL)
            .put("pinnedUntil",m.pinnedUntil ?: JSONObject.NULL).put("completedAt",m.completedAt ?: JSONObject.NULL).put("starArchiveAt",m.starArchiveAt ?: JSONObject.NULL).put("transcribed",m.transcribed)
            .put("reason", m.reason).put("manual", m.manual).put("sender",m.sender ?: JSONObject.NULL).put("attachment", m.attachment?.let { a -> JSONObject()
                .put("file", a.file).put("name", a.name).put("mime", a.mime).put("bytes", a.bytes).put("duration", a.duration).put("waveform",JSONArray(a.waveform)) } ?: JSONObject.NULL) })
        val report = state.report?.let { r -> JSONObject().put("confirmed", r.confirmed).put("heldOut", r.heldOut)
            .put("baseline", r.baseline).put("candidate", r.candidate).put("adopted", r.adopted).put("timestamp", r.timestamp) }
        val model = state.model?.let { m -> JSONObject().put("signature", m.signature)
            .put("weights", JSONArray(m.weights.map { JSONArray(it) })).put("bias", JSONArray(m.bias)) }
        return JSONObject().put("version", 1).put("categories", classes).put("messages", messages)
            .put("report", report ?: JSONObject.NULL).put("model", model ?: JSONObject.NULL).put("splits", JSONObject(state.splits))
            .put("preferences",JSONObject().put("preserveImportDates",state.preferences.preserveImportDates).put("appearance",state.preferences.appearance).put("widgetWhenLocked",state.preferences.widgetWhenLocked).put("classification",state.preferences.classification).put("frontend",state.preferences.frontend).put("autoTranslate",state.preferences.autoTranslate).put("targetLanguage",state.preferences.targetLanguage).put("ocr",state.preferences.ocr).put("reminders",state.preferences.reminders).put("includePrivateChatsInInbox",state.preferences.includePrivateChatsInInbox))
            .put("markdown",JSONObject().put("tree",state.markdown.tree ?: JSONObject.NULL).put("baseline",JSONObject(state.markdown.baseline)).put("synced",state.markdown.synced).put("pendingExternal",JSONObject(state.markdown.pendingExternal)).put("pendingRendered",JSONObject(state.markdown.pendingRendered)))
            .put("navigation",JSONObject().put("route",state.navigation.route).put("anchor",state.navigation.anchor ?: JSONObject.NULL).put("index",state.navigation.index).put("offset",state.navigation.offset).put("scroll",state.navigation.scroll))
            .put("chats",JSONArray(state.chats.map { JSONObject().put("id",it.id).put("title",it.title).put("icon",it.icon ?: JSONObject.NULL).put("photo",it.photo?.let(::attachmentJson) ?: JSONObject.NULL) }))
            .put("importedArchives",JSONArray(state.importedArchives.toList())).put("quiz",state.quiz?.let { q -> JSONObject().put("memoryKb",q.memoryKb).put("items",JSONArray(q.items.map { JSONObject().put("text",it.text).put("prediction",it.prediction ?: JSONObject.NULL).put("answer",it.answer ?: JSONObject.NULL).put("nanos",it.nanos) })) } ?: JSONObject.NULL)
            .put("editDraftId",state.editDraftId ?: JSONObject.NULL).put("preEditDraft",state.preEditDraft)
            .put("drafts",JSONObject(state.drafts.mapValues { (_,draft)->JSONObject().put("text",draft.text).put("replyTo",draft.replyTo ?: JSONObject.NULL).put("editId",draft.editId ?: JSONObject.NULL).put("previous",draft.previous).put("cursor",draft.cursor) }))
            .put("locations",JSONObject(state.locations.mapValues { navigation(it.value) }))
            .put("homeTab",state.homeTab).put("guide",JSONObject().put("returnTab",state.guide.returnTab).put("active",state.guide.active).put("step",state.guide.step).put("returnTo",navigation(state.guide.returnTo)))
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
                if(m.isNull("starArchiveAt"))null else m.getLong("starArchiveAt"),m.optBoolean("transcribed"),if(m.isNull("pinnedUntil"))null else m.getLong("pinnedUntil"),m.optional("chatId"),if(m.isNull("remindAt"))null else m.getLong("remindAt"),m.optional("translation"),m.optBoolean("extracted"))
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
        return ChatState(categories(root.getJSONArray("categories").toString()), messages, report, model, assignments,root.optString("draft"),root.optional("replyDraft"),root.optJSONObject("preferences")?.let { Preferences(it.optBoolean("preserveImportDates"),it.optString("appearance","system"),it.optBoolean("widgetWhenLocked"),it.optBoolean("classification",true),it.optString("frontend","chat"),it.optBoolean("autoTranslate"),it.optString("targetLanguage","en"),it.optBoolean("ocr"),it.optBoolean("reminders"),it.optBoolean("includePrivateChatsInInbox")) } ?: Preferences(),navigation,
            root.optJSONObject("markdown")?.let { md -> val files=md.optJSONObject("baseline"); MarkdownState(md.optional("tree"),files?.keys()?.asSequence()?.associateWith { files.getString(it) } ?: emptyMap(),md.optLong("synced"),md.optJSONObject("pendingExternal")?.let { values -> values.keys().asSequence().associateWith { values.getString(it) } } ?: emptyMap(),md.optJSONObject("pendingRendered")?.let { values -> values.keys().asSequence().associateWith { values.getString(it) } } ?: emptyMap()) } ?: MarkdownState(),
            root.optJSONArray("chats")?.let { rows -> (0 until rows.length()).map { i -> rows.getJSONObject(i).let { Chat(it.getString("id"),it.getString("title"),it.optional("icon"),it.optJSONObject("photo")?.let(::attachment)) } } } ?: emptyList(),
            root.optJSONArray("importedArchives")?.strings()?.toSet() ?: emptySet(),
            root.optJSONObject("quiz")?.let { q -> val rows=q.getJSONArray("items"); Quiz((0 until rows.length()).map { i -> rows.getJSONObject(i).let { QuizItem(it.getString("text"),it.optional("prediction"),it.optional("answer"),it.optLong("nanos")) } },q.optInt("memoryKb")) },root.optional("editDraftId"),root.optString("preEditDraft"),
            root.optJSONObject("drafts")?.let { rows->rows.keys().asSequence().associateWith { key->rows.getJSONObject(key).let { ComposerDraft(it.optString("text"),it.optional("replyTo"),it.optional("editId"),it.optString("previous"),it.optInt("cursor",-1)) } } } ?: emptyMap(),
            root.optJSONObject("locations")?.let { rows->rows.keys().asSequence().associateWith { navigation(rows.getJSONObject(it)) } } ?: emptyMap(),
            root.optJSONObject("guide")?.let { GuideProgress(it.optBoolean("active"),it.optInt("step").coerceIn(0,GuidePages.steps.lastIndex),it.optJSONObject("returnTo")?.let(::navigation) ?: Navigation("settings"),it.optString("returnTab","home").takeIf { tab->tab=="chats" } ?: "home") } ?: GuideProgress(),root.optString("homeTab","home").takeIf { it=="chats" } ?: "home")
    }
}
