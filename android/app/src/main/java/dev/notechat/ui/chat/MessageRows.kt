package dev.notechat

import android.content.Context
import android.view.ViewGroup
import android.widget.LinearLayout
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.RecyclerView

/** Updates message rows without rebuilding the screen or replacing its scroll manager. */
class MessageRows(private val context: Context, records: List<Message>, private val bind: (LinearLayout,Message,Int)->Unit) : RecyclerView.Adapter<MessageRows.Holder>() {
    private var records=records
    class Holder(val box: LinearLayout): RecyclerView.ViewHolder(box)
    init { setHasStableIds(true) }
    override fun getItemCount()=records.size
    override fun getItemId(position: Int)=records[position].id.fold(1125899906842597L) { hash,char -> hash*31+char.code }
    override fun onCreateViewHolder(parent: ViewGroup, type: Int)=Holder(LinearLayout(context).apply {
        orientation=LinearLayout.VERTICAL; layoutParams=RecyclerView.LayoutParams(-1,-2)
    })
    override fun onBindViewHolder(holder: Holder, position: Int) { holder.box.removeAllViews(); bind(holder.box,records[position],position) }
    fun replace(next: List<Message>) {
        val old=records
        val diff=DiffUtil.calculateDiff(object:DiffUtil.Callback() {
            override fun getOldListSize()=old.size
            override fun getNewListSize()=next.size
            override fun areItemsTheSame(a: Int,b: Int)=old[a].id==next[b].id
            override fun areContentsTheSame(a: Int,b: Int)=old[a]==next[b]
        })
        records=next; diff.dispatchUpdatesTo(this)
    }
    fun selectionChanged(ids: Set<String>) { records.forEachIndexed { index,record -> if(record.id in ids)notifyItemChanged(index) } }
}
