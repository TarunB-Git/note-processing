package dev.notechat

import org.json.JSONArray
import org.json.JSONObject

/** Shareable diagnostics. Personal examples, vectors, vocabulary and weights are excluded. */
object ModelReport {
    fun encode(state: ChatState, encoder: String?, memoryKb: Int=0): String {
        val confirmed=ChatEngine(state).confirmed()
        val evaluation=state.report?.let { JSONObject().put("heldOut",it.heldOut).put("baselineMacroF1",it.baseline)
            .put("candidateMacroF1",it.candidate).put("adopted",it.adopted) }
        return JSONObject().put("format","note-chat-model-report").put("version",1)
            .put("encoder",encoder ?: "built-in lexical features").put("dimensions",524)
            .put("features",JSONObject().put("sentenceEmbedding",384).put("wordingCues",12).put("hashedTokens",128))
            .put("activeHead",if(!state.preferences.classification)"disabled" else if(state.model?.let { TextClassifier.compatible(it,state.categories,encoder) }==true)"personal" else "starter").put("storedPersonalHead",state.model!=null).put("approvedCount",confirmed.size)
            .put("classes",JSONArray(state.categories.mapIndexed { index,c -> JSONObject().put("index",index).put("approvedCount",confirmed.count { it.category==c.id }) }))
            .put("classificationEnabled",state.preferences.classification).put("sessionMetrics",Diagnostics.report(memoryKb))
            .put("manualQuiz",state.quiz?.let { JSONObject().put("labelled",it.items.count { row->row.answer!=null }).put("total",it.items.size).put("accuracy",Diagnostics.accuracy(it) ?: JSONObject.NULL) } ?: JSONObject.NULL)
            .put("evaluation",evaluation ?: JSONObject.NULL)
            .put("excluded",JSONArray(listOf("messages","attachments","category names","category descriptions","examples","embeddings","trained weights","training split keys")))
            .toString(2)
    }
}
