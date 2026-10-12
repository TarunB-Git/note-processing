package dev.notechat

import android.content.Context
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.view.Gravity
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.GridLayoutManager
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.PagerSnapHelper
import androidx.recyclerview.widget.RecyclerView
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** A notes overview: one Inbox, followed by category and private-chat grids. */
class NoteBoard(
    context: Context, state: ChatState, private val colors: UiColors,
    private val renderer: MarkdownText, privateChatsOnly: Boolean = false, query: String = "",
    open: (String) -> Unit, manage: (String) -> Unit
) : RecyclerView(context) {
    private data class Tile(val route: String, val title: String, val records: List<Message>, val heading: Boolean = false, val wide: Boolean = false)
    private fun dp(value: Int) = (value * resources.displayMetrics.density).toInt()
    private fun label(value: String, size: Int, bold: Boolean = false) = TextView(context).apply {
        text = value
        textSize = size.toFloat()
        setTextColor(colors.ink)
        if (bold) setTypeface(null, Typeface.BOLD)
    }
    class Holder(val box: LinearLayout) : ViewHolder(box)
    init {
        setPadding(dp(10), 0, dp(10), dp(16))
        clipToPadding = false
        setBackgroundColor(colors.paper)
        itemAnimator = null
        val active = state.messages.filter { it.visible() }
        val inbox = active.filter { it.inInbox(state.preferences) }
        fun matches(title: String, notes: List<Message>) = query.isBlank() || title.contains(query,true) || notes.any { it.text.contains(query,true) || it.attachment?.name?.contains(query,true) == true }
        val categories = state.categories.sortedByDescending { it.pinned }.map { c -> Tile(c.id,c.title,active.filter { it.chatId==null && it.category==c.id }) } + Tile("unsorted","Unsorted",active.filter { it.chatId==null && !it.approved })
        val chats = state.chats.map { c -> Tile("chat:${c.id}",c.title,active.filter { it.chatId==c.id }) }
        val tiles = buildList {
            if(!privateChatsOnly) {
                if(matches("Inbox",inbox))add(Tile("all","Inbox",inbox,wide=true))
                add(Tile("","Categories",emptyList(),heading=true))
                addAll(categories.filter { matches(it.title,it.records) })
            }
            add(Tile("","Private chats",emptyList(),heading=true))
            addAll(chats.filter { matches(it.title,it.records) })
            if(chats.isEmpty())add(Tile("new-chat","Create a private chat",emptyList(),wide=true))
        }
        layoutManager = GridLayoutManager(context,2).apply {
            spanSizeLookup = object : GridLayoutManager.SpanSizeLookup() {
                override fun getSpanSize(position: Int) = if(tiles[position].wide || tiles[position].heading)2 else 1
            }
        }
        adapter = object : Adapter<Holder>() {
            override fun getItemCount() = tiles.size
            override fun onCreateViewHolder(parent: ViewGroup, viewType: Int) = Holder(LinearLayout(context).apply {
                orientation = LinearLayout.VERTICAL
                layoutParams = LayoutParams(-1,-2)
            })
            override fun onBindViewHolder(holder: Holder, position: Int) {
                val tile = tiles[position]
                holder.box.removeAllViews()
                holder.box.tag = "board:${tile.route}"
                holder.box.setPadding(dp(6),dp(6),dp(6),dp(6))
                if(tile.heading) {
                    holder.box.addView(label(tile.title,19,true).apply { setPadding(dp(2),dp(12),0,dp(6)) })
                    return
                }
                val card = LinearLayout(context).apply {
                    orientation = LinearLayout.VERTICAL
                    setPadding(dp(14),dp(14),dp(14),dp(12))
                    background = GradientDrawable().apply { setColor(this@NoteBoard.colors.surface); cornerRadius=dp(18).toFloat() }
                    contentDescription = "Open ${tile.title} board"
                    setOnClickListener { open(tile.route) }
                    setOnLongClickListener { manage(tile.route); true }
                }
                val heading=LinearLayout(context).apply { gravity=Gravity.CENTER_VERTICAL }
                heading.addView(ImageView(context).apply {
                    if(tile.route=="all")setImageDrawable(ChatIcon("notes",colors.accent))
                    else {
                        val icon=state.categories.firstOrNull { it.id==tile.route }?.icon ?: state.chats.firstOrNull { "chat:${it.id}"==tile.route }?.icon
                        setImageResource(CategoryIcons.resource(tile.route,icon)); setColorFilter(colors.accent)
                    }
                    importantForAccessibility=IMPORTANT_FOR_ACCESSIBILITY_NO
                },LinearLayout.LayoutParams(dp(24),dp(24)).apply { rightMargin=dp(8) })
                heading.addView(label(tile.title,if(tile.wide)20 else 16,true).apply {
                    maxLines=2
                    setOnClickListener { open(tile.route) }
                    setOnLongClickListener { manage(tile.route); true }
                },LinearLayout.LayoutParams(0,-2,1f))
                card.addView(heading)
                val recent = tile.records.asReversed().take(6)
                val modified=tile.records.maxOfOrNull { it.created }?.let { SimpleDateFormat("d MMM",Locale.getDefault()).format(Date(it)) }
                val total=resources.getQuantityString(R.plurals.note_count,tile.records.size,tile.records.size)
                val footer = label(if(tile.route=="new-chat")"Outside classification" else total,11).apply {
                    setTextColor(colors.muted)
                    setPadding(0,dp(10),0,0)
                    setOnClickListener { open(tile.route) }
                }
                if(recent.isEmpty())card.addView(label(if(tile.route=="new-chat")"Keep a separate notebook without category suggestions." else "Tap to start writing",14).apply {
                    setTextColor(colors.muted)
                    gravity=Gravity.CENTER_VERTICAL
                    minHeight=dp(if(tile.wide)68 else 104)
                    setOnClickListener { open(tile.route) }
                }) else {
                    val previews = RecyclerView(context).apply {
                        layoutManager = LinearLayoutManager(context,HORIZONTAL,false)
                        itemAnimator = null
                        overScrollMode = OVER_SCROLL_NEVER
                        contentDescription = "Swipe ${tile.title} preview"
                        adapter = object : Adapter<PreviewHolder>() {
                            override fun getItemCount()=recent.size
                            override fun onCreateViewHolder(parent: ViewGroup,type: Int)=PreviewHolder(MessageText(context) { open(tile.route) })
                            override fun onBindViewHolder(holder: PreviewHolder,index: Int) {
                                val note=recent[index]
                                holder.text.layoutParams=LayoutParams((width.takeIf { it>0 } ?: (resources.displayMetrics.widthPixels-dp(60))/if(tile.wide)1 else 2).coerceAtLeast(dp(90)),-1)
                                holder.text.textSize=14f
                                holder.text.setTextColor(colors.ink)
                                holder.text.maxLines=if(tile.wide)4 else 7
                                holder.text.ellipsize=android.text.TextUtils.TruncateAt.END
                                holder.text.setPadding(0,dp(10),dp(6),0)
                                renderer.render(holder.text,note.text.ifBlank { note.attachment?.name.orEmpty() },"",{})
                                holder.text.interceptTap={ open(tile.route); true }
                                holder.text.setOnClickListener { open(tile.route) }
                                holder.text.setOnLongClickListener { manage(tile.route); true }
                            }
                        }
                    }
                    val snap=PagerSnapHelper()
                    snap.attachToRecyclerView(previews)
                    previews.addOnLayoutChangeListener { _,left,_,right,_,oldLeft,_,oldRight,_ ->
                        if(right-left!=oldRight-oldLeft)previews.adapter?.notifyItemRangeChanged(0,recent.size)
                    }
                    previews.addOnScrollListener(object : OnScrollListener() {
                        override fun onScrollStateChanged(view: RecyclerView, state: Int) {
                            if(state==SCROLL_STATE_IDLE) {
                                val index=(previews.layoutManager as LinearLayoutManager).findFirstVisibleItemPosition().coerceAtLeast(0)
                                footer.text=context.getString(R.string.note_preview_page,total,index+1,recent.size)
                            }
                        }
                    })
                    card.addView(previews,LinearLayout.LayoutParams(-1,dp(if(tile.wide)88 else 104)))
                    if(recent.size>1)footer.text=context.getString(R.string.note_preview_swipe,total)
                }
                card.addView(footer)
                card.addView(label(modified.orEmpty(),10).apply { setTextColor(colors.muted); setPadding(0,dp(3),0,0); minHeight=dp(16) })
                holder.box.addView(card,LinearLayout.LayoutParams(-1,-2))
            }
        }
    }
    private class PreviewHolder(val text: MessageText) : ViewHolder(text)
}
