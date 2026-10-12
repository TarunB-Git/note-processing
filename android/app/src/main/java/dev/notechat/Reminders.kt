package dev.notechat

import android.app.AlarmManager
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import java.util.Calendar

object ReminderTime {
    fun detect(text: String, now: Long=System.currentTimeMillis()): Long? {
        val time=Regex("\\b([01]?\\d|2[0-3]):([0-5]\\d)\\s*(am|pm)?\\b|\\b(?:at\\s+)?(1[0-2]|[1-9])\\s*(am|pm)\\b",RegexOption.IGNORE_CASE).find(text) ?: return null
        val parts=time.groupValues
        var hour=(parts[1].ifBlank { parts[4] }).toInt()
        val minute=parts[2].ifBlank { "0" }.toInt(); val suffix=parts[3].ifBlank { parts[5] }.lowercase()
        if(suffix.isNotBlank()) { if(hour !in 1..12)return null; hour=hour%12+if(suffix=="pm")12 else 0 }
        val calendar=Calendar.getInstance().apply { timeInMillis=now; set(Calendar.HOUR_OF_DAY,hour); set(Calendar.MINUTE,minute); set(Calendar.SECOND,0); set(Calendar.MILLISECOND,0) }
        val date=Regex("\\b(\\d{4})-(\\d{2})-(\\d{2})\\b").find(text)
        if(date!=null) { calendar.isLenient=false; calendar.set(date.groupValues[1].toInt(),date.groupValues[2].toInt()-1,date.groupValues[3].toInt()) }
        else if(Regex("\\btomorrow\\b",RegexOption.IGNORE_CASE).containsMatchIn(text))calendar.add(Calendar.DAY_OF_YEAR,1)
        else if(calendar.timeInMillis<=now && !Regex("\\btoday\\b",RegexOption.IGNORE_CASE).containsMatchIn(text))calendar.add(Calendar.DAY_OF_YEAR,1)
        return runCatching { calendar.timeInMillis.takeIf { it>now } }.getOrNull()
    }
}

object Reminders {
    private const val channel="reminders"
    private fun intent(context: Context,id: String)=PendingIntent.getBroadcast(context,id.hashCode(),Intent(context,ReminderReceiver::class.java).setAction("dev.notechat.REMINDER").setData(android.net.Uri.parse("notechat://reminder/$id")).putExtra("id",id),PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
    fun infer(before: ChatState, state: ChatState): ChatState {
        if(!state.preferences.reminders)return state
        val old=before.messages.associateBy { it.id }
        return state.copy(messages=state.messages.map { message ->
            val previous=old[message.id]
            if(message.visible() && !message.done && message.category=="todos" && message.remindAt==null && (!before.preferences.reminders || previous==null || previous.text!=message.text || previous.category!=message.category))message.copy(remindAt=ReminderTime.detect(message.text)) else message
        })
    }
    fun sync(context: Context,state: ChatState) {
        val prefs=context.getSharedPreferences("reminder-times",Context.MODE_PRIVATE)
        val alarms=context.getSystemService(AlarmManager::class.java)
        val previous=prefs.all
        val next=if(state.preferences.reminders)state.messages.filter { it.visible() && !it.done && (it.remindAt ?: 0)>System.currentTimeMillis() }.associate { it.id to it.remindAt!! } else emptyMap()
        previous.keys.filter { it !in next || previous[it]!=next[it] }.forEach { alarms.cancel(intent(context,it)) }
        if(previous!=next)prefs.edit().clear().apply { next.forEach { (id,time)->putLong(id,time) } }.apply()
        next.filter { (id,time)->previous[id]!=time }.forEach { (id,time)->alarms.setAndAllowWhileIdle(AlarmManager.RTC_WAKEUP,time,intent(context,id)) }

    }
    fun reschedule(context: Context) {
        val alarms=context.getSystemService(AlarmManager::class.java)
        context.getSharedPreferences("reminder-times",Context.MODE_PRIVATE).all.forEach { (id,time) ->
            if(time is Long && time>System.currentTimeMillis())alarms.setAndAllowWhileIdle(AlarmManager.RTC_WAKEUP,time,intent(context,id))
        }
    }
    fun notify(context: Context,id: String) {
        val manager=context.getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(NotificationChannel(channel,"Reminders",NotificationManager.IMPORTANCE_DEFAULT))
        val launch=PendingIntent.getActivity(context,id.hashCode(),Intent(context,MainActivity::class.java).setData(android.net.Uri.parse("notechat://message/$id")).putExtra("jump_message",id),PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
        val note=Notification.Builder(context,channel).setSmallIcon(R.drawable.ic_note).setContentTitle("Note Chat reminder")
            .setContentText("Open your saved message").setAutoCancel(true).setContentIntent(launch).build()
        try { manager.notify(id.hashCode(),note) } catch(_:SecurityException) { }
        context.getSharedPreferences("reminder-times",Context.MODE_PRIVATE).edit().remove(id).apply()
    }
}
class ReminderReceiver: BroadcastReceiver() {
    override fun onReceive(context: Context,intent: Intent) {
        if(intent.action==Intent.ACTION_BOOT_COMPLETED)Reminders.reschedule(context)
        else intent.getStringExtra("id")?.let { id ->
            val time=context.getSharedPreferences("reminder-times",Context.MODE_PRIVATE).getLong(id,Long.MAX_VALUE)
            if(time<=System.currentTimeMillis())Reminders.notify(context,id)
        }
    }
}
