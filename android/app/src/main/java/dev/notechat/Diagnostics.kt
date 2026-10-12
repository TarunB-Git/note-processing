package dev.notechat

import org.json.JSONObject

object Diagnostics {
    private val samples=ArrayDeque<Long>()
    @Synchronized fun record(nanos: Long) { samples.addLast(nanos); if(samples.size>256)samples.removeFirst() }
    @Synchronized fun report(memoryKb: Int): JSONObject {
        val sorted=samples.sorted()
        fun percentile(value: Double)=if(sorted.isEmpty())JSONObject.NULL else sorted[((sorted.size-1)*value).toInt()]/1_000_000.0
        return JSONObject().put("sampleCount",sorted.size).put("medianClassificationMs",percentile(.5)).put("p95ClassificationMs",percentile(.95))
            .put("appPssKb",if(memoryKb>0)memoryKb else JSONObject.NULL).put("scope","Current session; classifier calls and whole-app PSS, not end-to-end UI latency")
    }
    fun accuracy(quiz: Quiz): Double? {
        val answered=quiz.items.filter { it.answer!=null }
        return if(answered.isEmpty())null else answered.count { it.prediction==it.answer }.toDouble()/answered.size
    }
}
