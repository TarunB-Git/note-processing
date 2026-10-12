package dev.notechat

import android.content.Context
import android.content.Intent
import android.content.pm.ShortcutInfo
import android.content.pm.ShortcutManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.drawable.Icon

object ChatShortcuts {
    fun pin(context: Context,route: String,title: String,resource: Int,photo: Bitmap? = null): Boolean {
        val manager=context.getSystemService(ShortcutManager::class.java)
        if(!manager.isRequestPinShortcutSupported)return false
        val icon=photo ?: Bitmap.createBitmap(128,128,Bitmap.Config.ARGB_8888).also { bitmap ->
            val canvas=Canvas(bitmap); canvas.drawColor(Color.rgb(217,253,211))
            androidx.appcompat.content.res.AppCompatResources.getDrawable(context,resource)?.mutate()?.apply { setTint(Color.rgb(0,128,105)); setBounds(24,24,104,104); draw(canvas) }
        }
        val shortcut=ShortcutInfo.Builder(context,"chat:$route").setShortLabel(title.take(40)).setLongLabel(title)
            .setIcon(Icon.createWithBitmap(icon)).setIntent(Intent(context,MainActivity::class.java).setAction(Intent.ACTION_VIEW).putExtra("chat_route",route)).build()
        return manager.requestPinShortcut(shortcut,null)
    }
}
