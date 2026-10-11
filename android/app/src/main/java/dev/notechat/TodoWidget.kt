package dev.notechat

import android.app.AlarmManager
import android.app.PendingIntent
import android.appwidget.AppWidgetManager
import android.appwidget.AppWidgetProvider
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.widget.RemoteViews
import android.widget.RemoteViewsService
import java.util.concurrent.Executors

class TodoWidget : AppWidgetProvider() {
    override fun onUpdate(context: Context, manager: AppWidgetManager, ids: IntArray) = refresh(context)
    override fun onReceive(context: Context, intent: Intent) {
        super.onReceive(context,intent)
        if(intent.action==REFRESH)refresh(context)
    }
    companion object {
        private const val REFRESH="dev.notechat.REFRESH_TODO"
        private val worker=Executors.newSingleThreadExecutor()
        fun refresh(context: Context) {
            val app=context.applicationContext
            worker.execute {
                val manager=AppWidgetManager.getInstance(app)
                val ids=manager.getAppWidgetIds(ComponentName(app,TodoWidget::class.java))
                if(ids.isEmpty())return@execute
                var locked=true
                var deadline:Long?=null
                try { Vault(app).useWorkspace { vault ->
                    locked=vault.lockedStorage
                    if(!locked) {
                        vault.openWithoutLock()
                        deadline=vault.load().messages.filter { it.deleted==null && it.done && it.completedAt!=null && !it.completedArchived() }.minOfOrNull { it.completedAt!!+ARCHIVE_DELAY_MS }
                    }
                } } catch(_:Exception) { locked=true }
                ids.forEach { id ->
                    val views=RemoteViews(app.packageName,R.layout.todo_widget)
                    views.setTextViewText(R.id.widget_title,if(locked)"Unlock Note Chat" else "To-do")
                    views.setViewVisibility(R.id.widget_list,if(locked)android.view.View.GONE else android.view.View.VISIBLE)
                    views.setViewVisibility(R.id.widget_empty,if(locked)android.view.View.GONE else android.view.View.VISIBLE)
                    views.setOnClickPendingIntent(R.id.widget_title,PendingIntent.getActivity(app,id,Intent(app,MainActivity::class.java).putExtra("widget_route","todos"),PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE))
                    val adapter=Intent(app,TodoWidgetService::class.java).putExtra(AppWidgetManager.EXTRA_APPWIDGET_ID,id).apply { data=android.net.Uri.parse("notechat://widget/$id") }
                    @Suppress("DEPRECATION") views.setRemoteAdapter(R.id.widget_list,adapter)
                    views.setEmptyView(R.id.widget_list,R.id.widget_empty)
                    // Fill-in data is restricted to this explicit activity and carries only a message ID.
                    views.setPendingIntentTemplate(R.id.widget_list,PendingIntent.getActivity(app,id+10000,Intent(app,MainActivity::class.java).putExtra("widget_route","todos"),PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_MUTABLE))
                    manager.updateAppWidget(id,views)
                    @Suppress("DEPRECATION") manager.notifyAppWidgetViewDataChanged(id,R.id.widget_list)
                }
                val alarms=app.getSystemService(AlarmManager::class.java)
                val alarm=PendingIntent.getBroadcast(app,0,Intent(app,TodoWidget::class.java).setAction(REFRESH),PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
                alarms.cancel(alarm)
                deadline?.let { alarms.setAndAllowWhileIdle(AlarmManager.RTC,it,alarm) }
            }
        }
    }
}

class TodoWidgetService : RemoteViewsService() {
    override fun onGetViewFactory(intent: Intent): RemoteViewsFactory = Tasks(applicationContext)
    private class Tasks(private val context:Context):RemoteViewsFactory {
        private var tasks=emptyList<Message>()
        override fun onCreate()=Unit
        override fun onDataSetChanged() {
            tasks=try { Vault(context).useWorkspace { vault ->
                if(vault.lockedStorage)emptyList() else { vault.openWithoutLock(); vault.load().messages.filter { it.visible() && it.category=="todos" } }
            } } catch(_:Exception) { emptyList() }
        }
        override fun onDestroy() { tasks=emptyList() }
        override fun getCount()=tasks.size
        override fun getViewAt(position:Int):RemoteViews {
            val views=RemoteViews(context.packageName,R.layout.todo_widget_row)
            val locked=try { Vault(context).useWorkspace { it.lockedStorage } } catch(_:Exception) { true }
            if(locked)return views
            val task=tasks.getOrNull(position) ?: return views
            views.setTextViewText(R.id.widget_text,task.text.ifBlank { task.attachment?.name.orEmpty() })
            views.setTextViewText(R.id.widget_check,if(task.done)"☑" else "☐")
            views.setOnClickFillInIntent(R.id.widget_check,Intent().putExtra("widget_complete",task.id))
            views.setOnClickFillInIntent(R.id.widget_text,Intent().putExtra("widget_jump",task.id))
            return views
        }
        override fun getLoadingView():RemoteViews?=null
        override fun getViewTypeCount()=1
        override fun getItemId(position:Int)=tasks.getOrNull(position)?.id?.hashCode()?.toLong() ?: position.toLong()
        override fun hasStableIds()=true
    }
}

private inline fun <T> Vault.useWorkspace(block:(Vault)->T):T = try { block(this) } finally { close() }
