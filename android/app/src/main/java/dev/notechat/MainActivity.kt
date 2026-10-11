package dev.notechat

import android.Manifest
import android.app.Activity
import androidx.appcompat.app.AlertDialog
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import android.app.Dialog
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.graphics.BitmapFactory
import android.os.Handler
import android.os.Looper
import android.os.Parcelable
import android.view.animation.DecelerateInterpolator
import com.google.android.material.bottomsheet.BottomSheetDialog
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.hardware.biometrics.BiometricManager
import android.hardware.biometrics.BiometricPrompt
import android.media.MediaPlayer
import android.media.MediaRecorder
import android.net.Uri
import android.os.Bundle
import android.os.CancellationSignal
import android.provider.OpenableColumns
import android.text.Editable
import android.text.TextWatcher
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.view.inputmethod.InputMethodManager
import android.widget.*
import androidx.core.content.FileProvider
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID
import java.util.concurrent.Executors
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import javax.crypto.Cipher

class MainActivity : Activity() {
    private val worker = Executors.newSingleThreadExecutor()
    private lateinit var vault: Vault
    private var engine: ChatEngine? = null
    private var encoder: MiniLm? = null
    private var classifier = TextClassifier(null)
    private var shown: ChatState? = null
    private var route = "home"
    private var query = ""
    private var reply: String? = null
    private var draft = ""
    private var busy = false
    private var resumed = false
    private var authInProgress = false
    private var dialogs = mutableListOf<Dialog>()
    private lateinit var root: LinearLayout
    private var list: RecyclerView? = null
    private var viewport: ChatViewport? = null
    private val positions = mutableMapOf<String,Navigation>()
    private var restoredNavigation: Navigation? = null
    private var highlightJump = false
    private var editor: EditText? = null
    private var pendingExport: File? = null
    private var pendingUri: Uri? = null
    private var pendingTextUri: Uri? = null
    private var pendingText: String? = null
    private var recorder: MediaRecorder? = null
    private var recordingFile: File? = null
    private var recordingStarted = 0L
    private var player: MediaPlayer? = null
    private var playbackFile: File? = null
    private var sessionGeneration = 0
    private val selected = linkedSetOf<String>()
    private var searchMode = false
    private var searchMatches = emptyList<Int>()
    private var searchCursor = -1
    private var searchCount: TextView? = null
    private var pendingJump: String? = null
    private var renderedRoute = ""
    private var overflowAnchor: View? = null
    private var activeOverflow: PopupMenu? = null
    private val handler = Handler(Looper.getMainLooper())
    private val voiceViews = mutableMapOf<String,VoiceMessageView>()
    private var playingId: String? = null
    private var playbackLoading = false
    private var playbackSpeed = 1f
    private val audioAttributes by lazy { android.media.AudioAttributes.Builder().setUsage(android.media.AudioAttributes.USAGE_MEDIA).setContentType(android.media.AudioAttributes.CONTENT_TYPE_SPEECH).build() }
    private val audioFocus by lazy { android.media.AudioFocusRequest.Builder(android.media.AudioManager.AUDIOFOCUS_GAIN_TRANSIENT).setAudioAttributes(audioAttributes).setWillPauseWhenDucked(true).setOnAudioFocusChangeListener { change->if(change<0)pausePlayback() }.build() }
    private fun acquireAudioFocus()=getSystemService(android.media.AudioManager::class.java).requestAudioFocus(audioFocus)==android.media.AudioManager.AUDIOFOCUS_REQUEST_GRANTED
    private fun pausePlayback() { if(!playbackLoading)try { player?.pause() } catch(_:Exception) { }; refreshVoiceViews() }
    private val noisyReceiver=object:android.content.BroadcastReceiver() { override fun onReceive(context:Context?,intent:Intent?) { pausePlayback() } }
    private var noisyRegistered=false
    private var pendingSeek = 0
    private val voiceDurations=mutableMapOf<String,Int>()
    private var recordingLocked = false
    private var recordingPaused = false
    private var pausedAt = 0L
    private var pausedDuration = 0L
    private val amplitudes = mutableListOf<Int>()
    private var composerContent: LinearLayout? = null
    private var recordButton: RecordButton? = null
    private var recordingClock: TextView? = null
    private var recordingWave: WaveformView? = null
    private var modelStatus: TextView? = null
    private var modelFailure: String? = null
    private var markdownMode: Boolean? = null
    private var markdownRenderer: MarkdownText? = null
    private val markdown: MarkdownText get() {
        if(markdownMode!=isDark()) { markdownMode=isDark(); markdownRenderer=MarkdownText(this,::openLink) }
        return markdownRenderer!!
    }
    private val imageCache=object:android.util.LruCache<String,android.graphics.Bitmap>(8*1024*1024) {
        override fun sizeOf(key:String,value:android.graphics.Bitmap)=value.byteCount
    }
    private val mediaTick = object:Runnable {
        override fun run() {
            if(recorder!=null) {
                if(!recordingPaused)try { amplitudes.add(recorder!!.maxAmplitude); if(amplitudes.size>3000)amplitudes.removeAt(0) } catch(_:Exception) { }
                recordingWave?.samples=amplitudes.takeLast(32)
                recordingClock?.text=duration(recordingElapsed())
            }
            refreshVoiceViews()
            if(recorder!=null || player!=null)handler.postDelayed(this,160)
        }
    }
    private val colors get() = UiColors(isDark())
    private val green get() = colors.accent
    private val paper get() = colors.paper
    private val ink get() = colors.ink
    private val muted get() = colors.muted
    private val bubble get() = colors.bubble
    private var splitTarget: String? = null
    private var splitPreviousDraft = ""
    private var splitPreviousReply: String? = null
    private var splitLines: LinearLayout? = null
    private var pendingWidgetToggle: String? = null
    private var pendingWidgetJump: String? = null
    private var autoUnlockAttempted = false
    private var imageDialog: Dialog? = null
    private var imageMessageId: String? = null
    private var renderedArchives=emptySet<String>()
    private fun archiveIds(state:ChatState)=state.messages.filter { it.deleted==null && (it.completedArchived() || it.starArchived()) }.map { it.id }.toSet()
    private val archiveTick = object : Runnable {
        override fun run() {
            if(resumed && !busy && shown!=null && recorder==null && splitTarget==null && !searchMode && query.isBlank() && selected.isEmpty() && archiveIds(shown!!)!=renderedArchives) {
                val focused=editor?.hasFocus()==true
                val cursor=editor?.selectionEnd ?: 0
                draft=editor?.text?.toString() ?: draft; render()
                if(focused)editor?.let { it.requestFocus(); it.setSelection(cursor.coerceIn(0,it.length())) }
            }
            if(resumed)handler.postDelayed(this,60_000)
        }
    }
    private var appearanceHint="system"
    private fun isDark() = when(shown?.preferences?.appearance ?: appearanceHint) {
        "dark" -> true; "light" -> false
        else -> resources.configuration.uiMode and Configuration.UI_MODE_NIGHT_MASK == Configuration.UI_MODE_NIGHT_YES
    }
    private fun applyAppearance() {
        shown?.preferences?.appearance?.let { appearanceHint=it; getSharedPreferences("display",MODE_PRIVATE).edit().putString("appearance",it).apply() }
        setTheme(if(isDark())R.style.AppThemeDark else R.style.AppTheme)
        root.setBackgroundColor(colors.surface)
        window.statusBarColor=colors.surface; window.navigationBarColor=colors.surface
        window.insetsController?.setSystemBarsAppearance(if(isDark())0 else android.view.WindowInsetsController.APPEARANCE_LIGHT_STATUS_BARS or android.view.WindowInsetsController.APPEARANCE_LIGHT_NAVIGATION_BARS,
            android.view.WindowInsetsController.APPEARANCE_LIGHT_STATUS_BARS or android.view.WindowInsetsController.APPEARANCE_LIGHT_NAVIGATION_BARS)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        vault = Vault(this)
        appearanceHint=getSharedPreferences("display",MODE_PRIVATE).getString("appearance","system") ?: "system"
        window.setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE)
        root = LinearLayout(this).apply { orientation=LinearLayout.VERTICAL; setBackgroundColor(colors.surface) }
        setContentView(root)
        root.setOnApplyWindowInsetsListener { view, insets ->
            val bars = insets.getInsets(android.view.WindowInsets.Type.systemBars())
            val keyboard = insets.getInsets(android.view.WindowInsets.Type.ime())
            view.setPadding(bars.left, bars.top, bars.right, maxOf(bars.bottom, keyboard.bottom))
            insets
        }
        if(android.os.Build.VERSION.SDK_INT>=33)registerReceiver(noisyReceiver,android.content.IntentFilter(android.media.AudioManager.ACTION_AUDIO_BECOMING_NOISY),Context.RECEIVER_NOT_EXPORTED) else { @Suppress("DEPRECATION") registerReceiver(noisyReceiver,android.content.IntentFilter(android.media.AudioManager.ACTION_AUDIO_BECOMING_NOISY)) }
        noisyRegistered=true
        savedInstanceState?.let { restoredNavigation=Navigation(it.getString("screen","home"),it.getString("anchor"),it.getInt("index"),it.getInt("offset"),it.getInt("scroll")) }
        captureIntent(intent)
        cacheDir.listFiles()?.filter { it.name.startsWith("record-") || it.name.startsWith("play-") }?.forEach { it.delete() }
        File(cacheDir,"shared").listFiles()?.filter { System.currentTimeMillis()-it.lastModified()>10*60*1000 }?.forEach { it.delete() }
        if (vault.lockedStorage) lockedScreen() else loadWorkspace()
    }
    override fun onNewIntent(intent: Intent) { super.onNewIntent(intent); setIntent(intent); captureIntent(intent); if (shown != null) processPending() }
    private fun captureIntent(intent: Intent?) {
        if(intent?.getStringExtra("widget_route")=="todos") {
            route="todos"; pendingWidgetToggle=intent.getStringExtra("widget_complete"); pendingWidgetJump=intent.getStringExtra("widget_jump")
            intent.removeExtra("widget_complete"); intent.removeExtra("widget_jump"); intent.removeExtra("widget_route")
        }
        if (intent?.action != Intent.ACTION_SEND) return
        pendingText = intent.getStringExtra(Intent.EXTRA_TEXT)
        @Suppress("DEPRECATION") pendingUri = intent.getParcelableExtra(Intent.EXTRA_STREAM)
    }
    override fun onResume() {
        super.onResume(); resumed=true
        if(vault.lockedStorage && shown==null && !authInProgress) { lockedScreen(); if(!autoUnlockAttempted) { autoUnlockAttempted=true; unlockWorkspace() } }
        else if(shown!=null && !vault.lockedStorage) {
            val generation=sessionGeneration
            worker.execute { val state=engine?.state; runOnUiThread {
                if(state!=null && resumed && generation==sessionGeneration) {
                    draft=editor?.text?.toString() ?: draft; shown=state; render(); processPending()
                }
            } }
        }
        handler.removeCallbacks(archiveTick); handler.postDelayed(archiveTick,60_000)
    }
    override fun onConfigurationChanged(configuration:Configuration) {
        super.onConfigurationChanged(configuration)
        draft=editor?.text?.toString() ?: draft
        if(shown!=null)render() else lockedScreen()
    }
    override fun onSaveInstanceState(outState: Bundle) {
        val location=capturePosition()
        outState.putString("screen",location.route); outState.putString("anchor",location.anchor)
        outState.putInt("index",location.index); outState.putInt("offset",location.offset); outState.putInt("scroll",location.scroll)
        super.onSaveInstanceState(outState)
    }
    override fun onStop() {
        super.onStop(); resumed=false; sessionGeneration++; autoUnlockAttempted=false; handler.removeCallbacks(archiveTick); cancelSplit(false); activeOverflow?.dismiss(); activeOverflow=null
        stopPlayback()
        if (recorder != null) stopRecording(send=false)
        val savedDraft=editor?.text?.toString() ?: draft
        val savedReply=reply
        val location=capturePosition(); positions[location.route]=location
        if(shown!=null)worker.execute { try { engine?.let { it.state=it.state.copy(draft=savedDraft,replyDraft=savedReply,navigation=location); vault.save(it.state) } } catch(_:Exception) { } }
        if (vault.lockedStorage && !authInProgress) {
            imageCache.evictAll()
            selected.clear(); searchMode=false
            draft=""; reply=null
            dialogs.toList().forEach { it.dismiss() }; dialogs.clear()
            shown=null; list=null; viewport=null; editor=null; root.removeAllViews(); lockedScreen()
            worker.execute { vault.close(); engine=null; encoder?.close(); encoder=null; classifier=TextClassifier(null) }
        }
    }
    override fun onDestroy() {
        super.onDestroy(); if(noisyRegistered) { unregisterReceiver(noisyReceiver); noisyRegistered=false }; handler.removeCallbacksAndMessages(null); stopPlayback(); if(recorder!=null) stopRecording(false)
        worker.execute { vault.close(); encoder?.close(); encoder=null }; worker.shutdown()
    }
    private fun loadWorkspace() {
        loading("Opening your chats…")
        worker.execute {
            try {
                if (!vault.isOpen) vault.openWithoutLock()
                val restored = ChatEngine(vault.load())
                val expired=restored.expire()
                restored.state=restored.state.copy(messages=restored.state.messages.map { if(it.done && it.completedAt==null)it.copy(completedAt=System.currentTimeMillis()) else it })
                vault.save(restored.state)
                TodoWidget.refresh(this)
                expired.mapNotNull { it.attachment }.forEach { vault.removeMedia(it) }
                engine=restored
                encoder?.close()
                encoder=try { if(MiniLm.modelFile(this).exists()) MiniLm(this) else null } catch(error:Exception) { modelFailure=error.message; null }
                classifier=TextClassifier(encoder)
                configure()
                val state=restored.state
                runOnUiThread { if (!isFinishing && (!vault.lockedStorage || resumed)) { shown=state; draft=state.draft; reply=state.replyDraft; busy=false; restoreLocation(state); privacy(); render(); processPending() } }
            } catch(error:Exception) { runOnUiThread { lockedScreen(); tell(error.message ?: "Could not open your workspace") } }
        }
    }
    private fun capturePosition(screen:String=renderedRoute.ifBlank { route }):Navigation {
        val recycler=list
        val manager=recycler?.layoutManager as? LinearLayoutManager
        val index=manager?.findFirstVisibleItemPosition()?.coerceAtLeast(0) ?: 0
        val item=manager?.findViewByPosition(index)
        val scrollView=(0 until root.childCount).map { root.getChildAt(it) }.filterIsInstance<ScrollView>().firstOrNull()
        val scroll=scrollView?.scrollY ?: 0
        return if(scrollView!=null)Navigation(screen,scroll=scroll) else if(item==null && list==null)positions[screen] ?: Navigation(screen)
            else Navigation(screen,item?.tag as? String,index,item?.top?.minus(recycler?.paddingTop ?: 0) ?: 0,scroll)
    }
    private fun restoreLocation(state:ChatState) {
        val location=restoredNavigation ?: if(renderedRoute.isBlank())state.navigation else positions[route]
        restoredNavigation=null
        if(location!=null) {
            if(route=="home" && renderedRoute.isBlank())route=location.route
            positions[location.route]=location
        }
        if(route !in setOf("home","all","starred","unsorted","settings","model","categories","deleted","completed","archived-stars") && state.categories.none { it.id==route })route="home"
    }
    private fun restorePosition(state:ChatState,position:Parcelable?) {
        val recycler=list
        val manager=recycler?.layoutManager as? LinearLayoutManager
        val location=positions[route]
        if(position!=null)manager?.onRestoreInstanceState(position)
        else if(location!=null && recycler!=null && manager!=null && (recycler.adapter?.itemCount ?: 0)>0) {
            val ids=when(route) {
                "deleted"->state.messages.filter { it.deleted!=null && System.currentTimeMillis()-it.deleted<RETENTION_MS }.sortedByDescending { it.deleted }.map { it.id }
                "completed"->state.messages.filter { it.deleted==null && it.completedArchived() }.map { it.id }
                "archived-stars"->state.messages.filter { it.deleted==null && it.starArchived() }.map { it.id }
                "home"->emptyList()
                else->messages(state,false).map { it.id }
            }
            val anchor=location.anchor?.let { ids.indexOf(it) } ?: -1
            manager.scrollToPositionWithOffset((if(anchor>=0)anchor else location.index).coerceIn(0,recycler.adapter!!.itemCount-1),location.offset)
        }
        if(location!=null)(0 until root.childCount).map { root.getChildAt(it) }.filterIsInstance<ScrollView>().firstOrNull()?.let { it.post { it.scrollTo(0,location.scroll) } }
        pendingJump?.let { id->
            pendingJump=null
            val highlighted=highlightJump; highlightJump=false
            if(viewport!=null)viewport!!.reveal(id,highlighted)
            else { val index=messages(state,false).indexOfFirst { it.id==id }; if(index>=0)manager?.scrollToPositionWithOffset(index,dp(30)) }
        }
    }
    private fun configure() { engine?.let { classifier.configure(it.state.categories,it.confirmed(),it.state.model,it.state.splits) } }
    private fun work(message:String="Saving…", refresh:Boolean=true, onFailure:(()->Unit)?=null, onSuccess:(()->Unit)?=null, action:()->Unit) {
        if(busy) { toast("Please wait for the current action to finish"); return }
        if(shown==null) return
        if(recorder!=null) { toast("Send or cancel the recording first"); return }
        val generation=sessionGeneration
        draft=editor?.text?.toString() ?: draft
        val storedDraft=draft
        val storedReply=reply
        busy=true
        val progress=ProgressBar(this,null,android.R.attr.progressBarStyleHorizontal).apply { isIndeterminate=true }
        root.addView(progress,0,LinearLayout.LayoutParams(-1,dp(3)))
        worker.execute {
            val before=engine?.state
            try {
                val current=engine ?: error("Unlock the app first")
                action()
                current.state=current.state.copy(draft=storedDraft,replyDraft=storedReply)
                vault.save(current.state)
                TodoWidget.refresh(this)
                val state=current.state
                runOnUiThread { root.removeView(progress); busy=false; if (!isFinishing && generation==sessionGeneration && (!vault.lockedStorage || resumed)) { shown=state; onSuccess?.invoke(); if(refresh) render(); if(message!="Saving…" && message.isNotBlank()) toast(message) } }
            } catch(error:Exception) {
                if(before!=null) { engine?.state=before; try { configure() } catch(_:Exception) { } }
                runOnUiThread { root.removeView(progress); busy=false; if(resumed) { onFailure?.invoke(); tell(error.message ?: "Could not complete this action") } }
            }
        }
    }
    private fun loading(text:String) { root.removeAllViews(); root.addView(label(text,17,muted).apply { gravity=Gravity.CENTER }, LinearLayout.LayoutParams(-1,-1)) }
    private fun lockedScreen() {
        applyAppearance()
        window.addFlags(WindowManager.LayoutParams.FLAG_SECURE)
        root.removeAllViews(); editor=null; list=null
        val content=column().apply { gravity=Gravity.CENTER; setPadding(dp(28),0,dp(28),0) }
        content.addView(ImageView(this).apply { setImageDrawable(ChatIcon("lock",green)) },LinearLayout.LayoutParams(dp(48),dp(48)))
        content.addView(label("Note Chat",28,ink,true).apply { gravity=Gravity.CENTER; setPadding(0,dp(24),0,dp(8)) })
        content.addView(label("Unlock to see your messages",15,muted).apply { gravity=Gravity.CENTER })
        val unlock=action("Unlock", ::unlockWorkspace).apply { setPadding(dp(30),dp(12),dp(30),dp(12)) }
        content.addView(unlock, LinearLayout.LayoutParams(-2,-2).apply { topMargin=dp(24) })
        root.addView(content,LinearLayout.LayoutParams(-1,-1))
    }
    private fun unlockWorkspace() {
        if(authInProgress)return
        try { authenticate(vault.unlockCipher()) { cipher ->
            worker.execute { try { vault.finishUnlock(cipher); runOnUiThread { loadWorkspace() } }
                catch(error:Exception) { runOnUiThread { tell(error.message ?: "Unlock failed") } } }
        } } catch(error:Exception) { tell(error.message ?: "Could not access the encryption key") }
    }
    private fun authenticate(cipher:Cipher, success:(Cipher)->Unit) {
        try {
            val allowed=BiometricManager.Authenticators.BIOMETRIC_STRONG or BiometricManager.Authenticators.DEVICE_CREDENTIAL
            val manager=getSystemService(BiometricManager::class.java)
            if(manager.canAuthenticate(allowed)!=BiometricManager.BIOMETRIC_SUCCESS) { tell("Set up a fingerprint or screen lock in Android settings first."); return }
            authInProgress=true
            val prompt=BiometricPrompt.Builder(this).setTitle("Unlock Note Chat").setSubtitle("Use your fingerprint or device screen lock")
                .setAllowedAuthenticators(allowed).build()
            prompt.authenticate(BiometricPrompt.CryptoObject(cipher),CancellationSignal(),mainExecutor,object:BiometricPrompt.AuthenticationCallback() {
                override fun onAuthenticationSucceeded(result:BiometricPrompt.AuthenticationResult) { authInProgress=false; success(result.cryptoObject?.cipher ?: return) }
                override fun onAuthenticationError(code:Int, text:CharSequence) { authInProgress=false; if(resumed) toast(text.toString()) }
            })
        } catch(error:Exception) { authInProgress=false; tell(error.message ?: "Authentication unavailable") }
    }
    private fun privacy() {
        if(vault.lockedStorage) window.addFlags(WindowManager.LayoutParams.FLAG_SECURE) else window.clearFlags(WindowManager.LayoutParams.FLAG_SECURE)
    }
    private fun processPending() {
        if(shown!=null && route!=renderedRoute)render()
        pendingWidgetToggle?.let { id -> pendingWidgetToggle=null
            val task=shown?.messages?.firstOrNull { it.id==id && it.visible() && it.category=="todos" }
            if(task!=null)work("") { engine!!.complete(id,!task.done) }
        }
        pendingWidgetJump?.let { id -> pendingWidgetJump=null; pendingJump=id; render() }
        pendingText?.let { draft=it.take(20000); pendingText=null; route="all"; render() }
        pendingUri?.let { uri -> pendingUri=null; val mime=contentResolver.getType(uri).orEmpty(); val name=contentResolver.query(uri,arrayOf(OpenableColumns.DISPLAY_NAME),null,null,null)?.use { if(it.moveToFirst())it.getString(0) else "" }.orEmpty(); if(mime.startsWith("text/") || name.endsWith(".md",true) || name.endsWith(".txt",true))importText(uri) else importUri(uri) }
        pendingTextUri?.let { uri->pendingTextUri=null; importText(uri) }
    }
    private fun navigate(destination:String) {
        cancelSplit(false); selected.clear(); searchMode=false; query=""; pendingJump=null
        if(destination!=route)stopPlayback()
        route=destination; render()
    }
    private fun render() {
        val state=shown ?: return
        applyAppearance()
        renderedArchives=archiveIds(state)
        if(renderedRoute.isNotBlank() && (list!=null || root.childCount>0))positions[renderedRoute]=capturePosition()
        val changed=route!=renderedRoute
        val position=if(!changed)list?.layoutManager?.onSaveInstanceState() else null
        val backwards=route=="home" || route=="settings"
        root.removeAllViews(); editor=null; list=null; viewport=null; overflowAnchor=null
        splitLines=null; composerContent=null; recordButton=null; recordingClock=null; recordingWave=null; modelStatus=null; voiceViews.clear()
        when(route) {
            "home" -> home(state)
            "settings" -> settings(state)
            "model" -> modelScreen(state)
            "categories" -> categories(state)
            "deleted" -> deleted(state)
            "completed", "archived-stars" -> archived(state)
            else -> conversation(state,position)
        }
        restorePosition(state,position)
        if(changed && renderedRoute.isNotEmpty() && android.animation.ValueAnimator.areAnimatorsEnabled())for(i in 0 until root.childCount)root.getChildAt(i).apply {
            alpha=0f; translationX=dp(if(backwards)-20 else 28).toFloat()
            animate().alpha(1f).translationX(0f).setDuration(190).setInterpolator(DecelerateInterpolator()).start()
        }
        renderedRoute=route
    }
    private fun home(state:ChatState) {
        val bar=row().apply { setPadding(dp(16),dp(8),dp(8),dp(8)) }
        bar.addView(label("Note Chat",25,green,true),LinearLayout.LayoutParams(0,dp(48),1f))
        bar.addView(icon("add","Add category") { editCategory(null) })
        bar.addView(icon("settings","Settings") { navigate("settings") })
        bar.addView(icon("more","More options") { homeMenu() }.also { overflowAnchor=it })
        root.addView(bar)
        searchBox()
        val active=state.messages.filter { it.visible() }
        val rows=mutableListOf(Triple("all","Inbox",active), Triple("starred","Starred",active.filter { it.starred && !it.starArchived() }))
        rows.addAll(state.categories.sortedByDescending { it.pinned }.map { c -> Triple(c.id,c.title,active.filter { it.category==c.id }) })
        rows.add(Triple("unsorted","Unsorted",active.filter { !it.approved }))
        fun filtered()=if(query.isBlank())rows else rows.filter { it.second.contains(query,true) }+active.filter { m->m.text.contains(query,true) || m.attachment?.name?.contains(query,true)==true }.map { m->Triple("message:${m.id}",state.categories.firstOrNull { it.id==m.category }?.title ?: "Inbox",listOf(m)) }
        val recycler=recycler()
        recycler.adapter=object:RecyclerView.Adapter<RowHolder>() {
            override fun getItemCount()=filtered().size
            override fun onCreateViewHolder(parent:ViewGroup,type:Int)=RowHolder(column())
            override fun onBindViewHolder(holder:RowHolder,position:Int) {
                val (id,title,messages)=filtered()[position]
                holder.box.removeAllViews()
                val item=row().apply { setPadding(dp(16),dp(10),dp(14),dp(10)); minimumHeight=dp(82) }
                val avatar=label(title.take(1),23,Color.WHITE,true).apply { gravity=Gravity.CENTER; background=rounded(if(id=="all")green else Color.rgb(96,151,130),28) }
                item.addView(avatar,LinearLayout.LayoutParams(dp(54),dp(54)).apply { rightMargin=dp(14) })
                val middle=column()
                middle.addView(label(title,17,ink,true))
                val last=messages.lastOrNull()
                middle.addView(label(last?.text?.ifBlank { last.attachment?.name.orEmpty() } ?: if(id=="all") "Send yourself a message" else "No messages yet",14,muted).apply { maxLines=1; ellipsize=android.text.TextUtils.TruncateAt.END; setPadding(0,dp(6),0,0) })
                item.addView(middle,LinearLayout.LayoutParams(0,-2,1f))
                val right=column().apply { gravity=Gravity.END }
                right.addView(label(last?.let { time(it.created) } ?: "",11,muted))
                if(!id.startsWith("message:") && id !in listOf("all","starred","unsorted") && messages.isNotEmpty()) right.addView(label(messages.size.toString(),11,muted).apply { setTypeface(null,Typeface.ITALIC); setPadding(0,dp(6),0,0) })
                item.addView(right)
                item.setOnClickListener { if(id.startsWith("message:")) { route="all"; selected.clear(); searchMode=true; val current=this@MainActivity.messages(state,false); searchMatches=matching(current); searchCursor=searchMatches.indexOf(current.indexOfFirst { it.id==id.removePrefix("message:") }); render() } else navigate(id) }
                if(state.categories.any { it.id==id }) item.setOnLongClickListener { categoryMenu(state.categories.first { it.id==id }); true }
                holder.box.addView(item)
                holder.box.addView(View(this@MainActivity).apply { setBackgroundColor(colors.divider) },LinearLayout.LayoutParams(-1,dp(1)).apply { leftMargin=dp(84) })
            }
        }
        list=recycler; root.addView(recycler,LinearLayout.LayoutParams(-1,0,1f))
    }
    private fun homeMenu() { overflow(listOf("Import TXT / Markdown","Reclassify all","Manage categories","Settings")) { when(it) { 0->pickText(); 1->reclassifyAll(); 2->navigate("categories"); 3->navigate("settings") } } }
    private fun messages(state:ChatState,includeSearch:Boolean=true)=state.messages.filter { it.visible() && when(route) {
        "all" -> true; "starred" -> it.starred && !it.starArchived(); "unsorted" -> !it.approved; else -> it.category==route
    } }.filter { !includeSearch || query.isBlank() || it.text.contains(query,true) || it.attachment?.name?.contains(query,true)==true }
    private fun title(state:ChatState)=when(route) { "all"->"Inbox"; "starred"->"Starred"; "unsorted"->"Unsorted"; else->state.categories.firstOrNull { it.id==route }?.title ?: "Inbox" }
    private fun conversation(state:ChatState,position:Parcelable?=null) {
        if(route=="todos" && !searchMode) { todoList(state); return }
        val current=messages(state,false)
        selected.retainAll(current.map { it.id }.toSet())
        if(selected.isNotEmpty()) selectionToolbar(state)
        else if(searchMode) searchToolbar(current)
        else toolbar(title(state),"",{ navigate("home") },listOf("search" to { beginSearch() },"more" to { chatMenu() }))
        val recycler=recycler().apply { setBackgroundColor(paper) }
        val manager=recycler.layoutManager as LinearLayoutManager
        manager.stackFromEnd=true
        recycler.adapter=object:RecyclerView.Adapter<RowHolder>() {
            override fun getItemCount()=current.size
            override fun onCreateViewHolder(parent:ViewGroup,type:Int)=RowHolder(column())
            override fun onBindViewHolder(holder:RowHolder,position:Int) {
                holder.box.removeAllViews()
                val message=current[position]
                holder.box.setBackgroundColor(if(message.id in selected)colors.selected else Color.TRANSPARENT)
                if(position==0 || day(message.created)!=day(current[position-1].created)) {
                    val date=row().apply { gravity=Gravity.CENTER; setPadding(0,dp(8),0,dp(8)) }
                    date.addView(pill(day(message.created),muted,colors.inset)); holder.box.addView(date)
                }
                holder.box.addView(messageBubble(state,message))
                viewport?.bind(holder.box,message.id)
            }
        }
        list=recycler; viewport=ChatViewport(this,recycler,current.map { it.id },colors)
        root.addView(viewport,LinearLayout.LayoutParams(-1,0,1f))
        if(position!=null)manager.onRestoreInstanceState(position) else if(current.isNotEmpty())recycler.scrollToPosition(current.lastIndex)
        if(current.isEmpty())root.addView(label(if(route=="unsorted")"No messages awaiting approval" else "Send yourself a message",13,muted).apply { gravity=Gravity.CENTER; setPadding(dp(16),dp(8),dp(16),dp(8)) })
        if(searchMode)searchNavigation(current)
        if(selected.isEmpty())composer(state)
    }
    private fun messageBubble(state:ChatState,message:Message,recovering:Boolean=false):View {
        val width=(resources.displayMetrics.widthPixels-dp(60)).coerceAtLeast(dp(180))
        val wrapper=ReplySwipe(this@MainActivity) { if(!recovering)replyTo(message) }.apply {
            setPadding(dp(44),dp(3),dp(8),dp(3)); enabledForReply=!recovering && selected.isEmpty()
            layoutParams=LinearLayout.LayoutParams(-1,-2)
        }
        val body=BubbleLayout(this@MainActivity,width).apply { orientation=LinearLayout.VERTICAL; background=rounded(bubble,10); setPadding(dp(9),dp(6),dp(9),dp(4)); minimumWidth=dp(150) }
        val quote=message.replyTo?.let { id->state.messages.firstOrNull { it.id==id && it.deleted==null } }
        if(message.replyTo!=null)body.addView(replyPreview(quote).apply {
            if(!recovering)setOnClickListener { if(selected.isNotEmpty())toggleSelection(message.id) else quote?.let { jumpTo(it.id,true) } }
            if(!recovering)setOnLongClickListener { toggleSelection(message.id); true }
        },LinearLayout.LayoutParams(-1,-2).apply { bottomMargin=dp(4) })
        message.attachment?.let { a -> when {
            a.mime.startsWith("image/")->imageBubble(body,message,width-dp(18),recovering)
            a.mime.startsWith("audio/")-> {
                val voice=VoiceMessageView(this@MainActivity,a.waveform,{ if(selected.isNotEmpty())toggleSelection(message.id) else playVoice(message) },{ seekVoice(message,it) },{ changeSpeed() })
                voice.selectionMode=selected.isNotEmpty()
                voiceViews[message.id]=voice
                body.addView(voice,LinearLayout.LayoutParams(width-dp(18),-2))
                if(!recovering)voice.setOnLongClickListener { toggleSelection(message.id); true }
                updateVoice(message.id,voice)
            }
            else->{ val media=row(); media.addView(icon("attach","Open ${a.name}") { if(selected.isNotEmpty())toggleSelection(message.id) else openAttachment(message) }); media.addView(label(a.name,15,ink).apply { maxLines=2 },LinearLayout.LayoutParams(0,-2,1f)); body.addView(media,LinearLayout.LayoutParams(width-dp(18),-2)) }
        } }
        if(message.text.isNotBlank())body.addView(MessageText(this@MainActivity) { if(!recovering)tagging(message) }.apply {
            textSize=16f; setTextColor(ink); setPadding(0,dp(3),0,dp(5)); maxWidth=width-dp(18)
            interceptTap={ if(selected.isNotEmpty()) { toggleSelection(message.id); true } else false }
            linkMenu=::linkActions
            markdown.render(this,message.text,if(searchMode && !recovering)query else "",::openLink)
            setOnClickListener { if(selected.isNotEmpty())toggleSelection(message.id) }
            if(!recovering)setOnLongClickListener { toggleSelection(message.id); true }
        })
        val footer=row().apply { gravity=Gravity.CENTER_VERTICAL }
        val category=state.categories.firstOrNull { it.id==message.category }?.title ?: "Tag"
        footer.addView(label(category,11,green).apply { setPadding(0,dp(4),dp(10),dp(4)); if(!recovering)setOnClickListener { if(selected.isNotEmpty())toggleSelection(message.id) else tagging(message) } },LinearLayout.LayoutParams(0,-2,1f))
        footer.addView(label("${if(message.starred)"★ " else ""}${if(message.edited)"edited · " else ""}${time(message.created)}${if(message.approved)" ✓✓" else " ✓"}",10,muted))
        body.addView(footer)
        if(recovering) {
            val days=((RETENTION_MS-(System.currentTimeMillis()-message.deleted!!))/(24*60*60*1000)).coerceAtLeast(0)
            body.addView(label("Deleted ${day(message.deleted)} · $days days left",11,muted).apply { setPadding(0,dp(4),0,0) })
            body.addView(row().apply { gravity=Gravity.END; addView(action("Restore",{ restoreMessage(message) },false)) })
        }
        val feedback=row().apply { gravity=Gravity.END }
        if(!recovering && message.category!=null) {
            feedback.addView(action(if(message.approved)"👍 ✓" else "👍",{ work("") { engine!!.approve(message.id); configure(); maybeTrain() } },false).apply { contentDescription=if(message.trainable())"Approve category and teach the classifier" else "Approve category" })
            if(message.trainable())feedback.addView(action("👎",{ work("") { val current=engine!!; classifier.configure(current.state.categories,current.confirmed().filterNot { it.id==message.id },if(message.approved)null else current.state.model,current.state.splits); current.reject(message.id) { classifier.classify(it) }; configure() } },false).apply { contentDescription="Try another category without learning" })
        }
        if(feedback.childCount>0 && selected.isEmpty())body.addView(feedback)
        body.setOnClickListener { if(selected.isNotEmpty())toggleSelection(message.id) }
        if(!recovering)body.setOnLongClickListener { toggleSelection(message.id); true }
        wrapper.addView(body,FrameLayout.LayoutParams(-2,-2,Gravity.END))
        return wrapper
    }
    private fun composer(state:ChatState) {
        val box=column().apply { setBackgroundColor(paper); setPadding(dp(7),dp(5),dp(7),dp(7)) }
        if(splitTarget!=null)splitControls(box)
        if(reply!=null) {
            val previous=state.messages.firstOrNull { it.id==reply && it.deleted==null }
            box.addView(replyPreview(previous) { reply=null; render() },LinearLayout.LayoutParams(-1,-2).apply { leftMargin=dp(4); rightMargin=dp(54); bottomMargin=dp(4) })
        }
        val line=row()
        val content=column(); composerContent=content
        line.addView(content,LinearLayout.LayoutParams(0,-2,1f))
        val button=RecordButton(this).apply {
            background=rounded(green,24); setPadding(dp(12),dp(12),dp(12),dp(12)); contentDescription="Send message. Hold microphone to record; slide left to cancel or up to lock."
            setOnClickListener { if(recorder!=null)stopRecording(true) else if(draft.isNotBlank()) { if(splitTarget!=null)sendSplitPart() else sendText(draft) } else if(recordVoice()) { recordingLocked=true; updateRecordingComposer() } }
            gesturesEnabled=draft.isBlank() && recorder==null && splitTarget==null
            startHold={ recordVoice() }; releaseHold={ stopRecording(true) }; cancelHold={ stopRecording(false) }; lockHold={ recordingLocked=true; updateRecordingComposer() }
        }
        recordButton=button
        line.addView(button,LinearLayout.LayoutParams(dp(48),dp(48)).apply { leftMargin=dp(6) }); box.addView(line); root.addView(box)
        if(recorder!=null)updateRecordingComposer() else {
            val inputRow=row().apply { background=rounded(colors.surface,26); setPadding(dp(4),0,dp(3),0) }
            inputRow.addView(icon("attach","Attach or import") { attachMenu() }.apply { isEnabled=splitTarget==null; alpha=if(isEnabled)1f else .35f })
            val input=EditText(this).apply { hint="Message"; textSize=16f; setTextColor(ink); setHintTextColor(muted); background=null; setText(draft); maxLines=5; setPadding(dp(4),dp(9),dp(4),dp(9)) }
            editor=input; inputRow.addView(input,LinearLayout.LayoutParams(0,-2,1f)); inputRow.addView(icon("camera","Choose image") { pickImage() }.apply { isEnabled=splitTarget==null; alpha=if(isEnabled)1f else .35f }); content.addView(inputRow)
            input.addTextChangedListener(watcher { draft=it; if(splitTarget!=null)updateSplitLines(it); button.gesturesEnabled=it.isBlank() && splitTarget==null; button.setImageDrawable(ChatIcon(if(it.isNotBlank())"send" else "mic",Color.WHITE)) })
            button.setImageDrawable(ChatIcon(if(draft.isNotBlank())"send" else "mic",Color.WHITE))
        }
    }
    private fun updateRecordingComposer() {
        val content=composerContent ?: return
        content.removeAllViews(); editor=null
        val recording=row().apply { background=rounded(colors.surface,24); setPadding(dp(5),0,dp(5),0) }
        if(recordingLocked)recording.addView(icon("delete","Discard recording") { stopRecording(false) })
        else recording.addView(label("●",18,Color.rgb(220,67,80)).apply { setPadding(dp(8),0,dp(5),0) })
        recordingClock=label(duration(recordingElapsed()),14,ink).also { recording.addView(it) }
        recordingWave=WaveformView(this).apply { live=true; samples=amplitudes.takeLast(32) }.also { recording.addView(it,LinearLayout.LayoutParams(0,dp(44),1f)) }
        if(recordingLocked)recording.addView(icon(if(recordingPaused)"mic" else "pause",if(recordingPaused)"Resume recording" else "Pause recording") { pauseRecording() })
        content.addView(recording)
        if(!recordingLocked)content.addView(label("‹ Slide to cancel · ↑ lock",11,muted).apply { gravity=Gravity.CENTER; setPadding(0,dp(3),0,dp(3)) })
        recordButton?.setImageDrawable(ChatIcon(if(recordingLocked)"send" else "mic",Color.WHITE))
        if(recordingLocked)recordButton?.gesturesEnabled=false
    }
    private fun sendText(text:String) {
        if(busy) { toast("Please wait for the current action to finish"); return }
        if(text.length>20000) { tell("Messages are limited to 20,000 characters."); return }
        val category=shown?.categories?.firstOrNull { it.id==route }?.id
        val replyId=reply; draft=""; editor?.setText(""); reply=null
        work("",onFailure={ draft=text; reply=replyId; render() }) {
            val current=engine!!
            val message=Message(text=text.trim(),replyTo=replyId)
            val suggestion=if(category!=null)Suggestion(category,"Sent to this category. Tap thumbs-up to teach it.") else classifier.classify(message)
            current.state=current.state.copy(messages=current.state.messages + message.copy(category=suggestion.category,reason=suggestion.reason,manual=category!=null))
        }
    }
    private fun toggleSelection(id:String) {
        if(id in selected)selected.remove(id) else selected.add(id)
        searchMode=false; query=""; render()
    }
    private fun selectionToolbar(state:ChatState) {
        val bar=row().apply { setBackgroundColor(colors.surface) }
        bar.addView(icon("back","Clear selection") { selected.clear(); render() })
        bar.addView(label(selected.size.toString(),20,ink,true),LinearLayout.LayoutParams(0,-2,1f))
        if(selected.size==1)bar.addView(icon("reply","Reply") { state.messages.firstOrNull { it.id in selected }?.let { replyTo(it) } })
        bar.addView(icon("star","Star selected messages") { val ids=selected.toSet(); selected.clear(); work("") { ids.forEach { id->engine!!.update(id) { it.copy(starred=true,starArchiveAt=null) } } } })
        bar.addView(icon("copy","Copy selected messages") { copyText(state.messages.filter { it.id in selected }.joinToString("\n\n") { it.text.ifBlank { it.attachment?.name.orEmpty() } }); selected.clear(); render() })
        bar.addView(icon("delete","Delete selected messages") { confirmDelete(selected.toSet()) })
        bar.addView(icon("more","Selected message options") { selectionMenu(state) }.also { overflowAnchor=it })
        root.addView(bar)
    }
    private fun selectionMenu(state:ChatState) {
        val rows=state.messages.filter { it.id in selected }
        val options=linkedMapOf<String,()->Unit>()
        options["Tag messages"]={ tagSelection() }
        options["Share"]={ if(rows.size==1 && rows[0].attachment!=null)shareAttachment(rows[0]) else if(rows.any { it.attachment!=null })shareSelection(rows) else shareText(rows.joinToString("\n\n") { it.text }) }
        options["Select all"]={ selected.addAll(messages(state,false).map { it.id }); render() }
        options["Unstar"]={ val ids=selected.toSet(); selected.clear(); work("") { ids.forEach { id->engine!!.update(id) { it.copy(starred=false,starArchiveAt=null) } } } }
        if(rows.any { it.starred })options["Archive starred in 24 hours"]={ val ids=rows.filter { it.starred }.map { it.id }.toSet(); selected.clear(); work("") { engine!!.archiveStars(ids) } }
        if(rows.size==1) {
            val message=rows.first()
            options["Edit"]={ selected.clear(); editMessage(message) }
            if(message.attachment==null)options["Split message"]={ startSplit(message) }
            if(message.attachment?.mime?.startsWith("audio/")==true)options["Transcribe voice message"]={ selected.clear(); transcribe(message) }
            options["Why this tag?"]={ tell(message.reason.ifBlank { "Manually tagged attachment." }) }
        } else if(rows.size>1 && rows.all { it.attachment==null })options["Merge messages"]={
            dialog("Merge ${rows.size} messages?").setMessage("Keep the first message's position and time. Replies will point to the merged message. The combined text gets a new category suggestion.")
                .setNegativeButton("Cancel",null).setPositiveButton("Merge") { _,_-> val ids=rows.map { it.id }.toSet(); selected.clear(); work("") { val merged=engine!!.merge(ids); configure(); suggest(merged); pendingJump=merged.id } }.showTracked()
        }
        overflow(options.keys.toList()) { options.values.elementAt(it)() }
    }
    private fun confirmDelete(ids:Set<String>) {
        dialog("Delete ${ids.size} ${if(ids.size==1)"message" else "messages"}?").setMessage("Restore from Recently deleted for 90 days.")
            .setNegativeButton("Cancel",null).setPositiveButton("Delete") { _,_->selected.clear(); if(playingId in ids)stopPlayback(); if(imageMessageId in ids)imageDialog?.dismiss(); work("") { engine!!.delete(ids); configure() } }.showTracked()
    }
    private fun replyPreview(message:Message?,close:(()->Unit)?=null)=ReplyPreview(this,message,colors,close).apply {
        if(message?.text?.isNotBlank()==true)markdown.render(summary,message.text.take(300),"",::openLink)
        summary.movementMethod=null; summary.isClickable=false; summary.isLongClickable=false
        photo?.let { image->message?.let { loadImage(it,image) } }
    }
    private fun replyTo(message:Message) { selected.clear(); reply=message.id; render(); editor?.requestFocus(); (getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager).showSoftInput(editor,InputMethodManager.SHOW_IMPLICIT) }
    private fun jumpTo(id:String,highlight:Boolean=false) { if(messages(shown!!,false).none { it.id==id })route="all"; pendingJump=id; highlightJump=highlight; render() }
    private fun tagging(m:Message) { chooseTags(setOf(m.id)) }
    private fun tagSelection() { chooseTags(selected.toSet()) }
    private fun chooseTags(ids:Set<String>) {
        val categories=shown?.categories ?: return
        menu("Tag ${if(ids.size==1)"message" else "messages"}",categories.map { it.title }+"Unsorted") { i-> selected.clear(); work("") { ids.forEach { engine!!.tag(it,categories.getOrNull(i)?.id) }; configure() } }
    }
    private fun editMessage(m:Message) {
        val input=EditText(this).apply { setText(m.text); maxLines=8 }
        dialog("Edit message").setView(input).setNegativeButton("Cancel",null).setPositiveButton("Save") { _,_->
            val text=input.text.toString()
            work("") { engine!!.edit(m.id,text); configure(); val note=engine!!.state.messages.first { it.id==m.id }; if(note.trainable())suggest(note) }
        }.showTracked()
    }
    private fun chatMenu() { overflow(listOf("Search","Import TXT / Markdown","Select messages","Export chat","Manage categories","Delete chat messages","Go to first message")) { when(it) {
        0->beginSearch(); 1->pickText(); 2->{ messages(shown!!,false).lastOrNull()?.let { m->toggleSelection(m.id) } }
        3->exportMenu(); 4->navigate("categories"); 5->confirmDelete(messages(shown!!,false).map { it.id }.toSet()); 6->messages(shown!!,false).firstOrNull()?.let { jumpTo(it.id) }
    } } }
    private fun beginSearch() { selected.clear(); searchMode=true; query=""; searchCursor=-1; render() }
    private fun searchToolbar(current:List<Message>) {
        val bar=row()
        bar.addView(icon("back","Close search") { searchMode=false; query=""; render() })
        val input=EditText(this).apply { hint="Search…"; textSize=17f; background=null; setSingleLine(); setText(query) }
        bar.addView(input,LinearLayout.LayoutParams(0,dp(52),1f))
        bar.addView(icon("close","Clear search") { input.setText("") })
        root.addView(bar)
        input.addTextChangedListener(watcher { query=it; searchMatches=matching(current); searchCursor=searchMatches.lastIndex; refreshSearch(current,true) })
        input.setOnEditorActionListener { _,_,_->(getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager).hideSoftInputFromWindow(input.windowToken,0); true }
        input.requestFocus(); input.post { if(resumed && searchMode)(getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager).showSoftInput(input,InputMethodManager.SHOW_IMPLICIT) }
    }
    private fun matching(current:List<Message>)=if(query.isBlank())emptyList() else current.indices.filter { current[it].text.contains(query,true) || current[it].attachment?.name?.contains(query,true)==true }
    private fun searchNavigation(current:List<Message>) {
        searchMatches=matching(current); if(searchCursor !in searchMatches.indices)searchCursor=searchMatches.lastIndex
        val bar=row().apply { setBackgroundColor(colors.surface); setPadding(dp(16),0,dp(4),0) }
        searchCount=label("",13,muted).also { bar.addView(it,LinearLayout.LayoutParams(0,-2,1f)) }
        bar.addView(icon("up","Previous search result") { if(searchCursor>0) { searchCursor--; refreshSearch(current,false) } })
        bar.addView(icon("down","Next search result") { if(searchCursor<searchMatches.lastIndex) { searchCursor++; refreshSearch(current,false) } })
        root.addView(bar); refreshSearch(current,false)
    }
    private fun refreshSearch(current:List<Message>,rebind:Boolean) {
        searchCount?.text=if(query.isBlank())"Search this chat" else if(searchMatches.isEmpty())"No results" else "${searchCursor+1} of ${searchMatches.size}"
        if(rebind && current.isNotEmpty())list?.adapter?.notifyItemRangeChanged(0,current.size,"search")
        searchMatches.getOrNull(searchCursor)?.let { (list?.layoutManager as? LinearLayoutManager)?.scrollToPositionWithOffset(it,dp(30)) }
    }
    private fun openLink(value:String) {
        val link=if(value.contains(":") && !value.startsWith("www."))value else "https://$value"
        val uri=Uri.parse(link)
        if(uri.scheme !in listOf("https","http","mailto","tel")) { tell("This link type cannot be opened"); return }
        try { startActivity(Intent(Intent.ACTION_VIEW,uri)) } catch(_:Exception) { tell("No app can open this link") }
    }
    private fun linkActions(link:String) { menu(link,listOf("Open link","Copy link","Share link")) { when(it) { 0->openLink(link); 1->copyText(link); 2->shareText(link) } } }
    private fun copyText(text:String) { (getSystemService(CLIPBOARD_SERVICE) as ClipboardManager).setPrimaryClip(ClipData.newPlainText("Message",text)) }
    private fun shareText(text:String) { startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).apply { type="text/plain"; putExtra(Intent.EXTRA_TEXT,text) },"Share")) }
    private fun watcher(changed:(String)->Unit)=object:TextWatcher { override fun beforeTextChanged(s:CharSequence?,start:Int,count:Int,after:Int)=Unit; override fun onTextChanged(s:CharSequence?,start:Int,before:Int,count:Int) { changed(s.toString()) }; override fun afterTextChanged(s:Editable?)=Unit }
    private fun settings(state:ChatState) {
        toolbar("Settings","Local workspace",{ navigate("home") })
        val scroll=ScrollView(this); val content=column().apply { setPadding(dp(16),dp(12),dp(16),dp(24)) }; scroll.addView(content); root.addView(scroll,LinearLayout.LayoutParams(-1,0,1f))
        setting(content,"App lock",if(vault.lockedStorage)"Fingerprint or device screen lock · locks when you leave" else "Off · messages and attachments are still encrypted") {
            if(vault.lockedStorage) dialog("Turn off app lock?").setMessage("Your device-encrypted storage remains, but opening the app will no longer ask for authentication.")
                .setNegativeButton("Cancel",null).setPositiveButton("Turn off") { _,_->work { vault.disableLock(); runOnUiThread { privacy() } } }.showTracked()
            else try {
                authenticate(vault.enableLockCipher()) { cipher -> work("App lock enabled") { vault.finishWrapping(cipher,true); runOnUiThread { privacy() } } }
            } catch(_:Exception) { tell("Set up a fingerprint or device screen lock in Android settings first.") }
        }
        setting(content,"Appearance",state.preferences.appearance.replaceFirstChar { it.uppercase() }) {
            menu("Appearance",listOf("System","Light","Dark")) { index -> work("") { engine!!.state=engine!!.state.copy(preferences=engine!!.state.preferences.copy(appearance=listOf("system","light","dark")[index])) } }
        }
        setting(content,"Keep imported dates",if(state.preferences.preserveImportDates)"On · use transcript dates" else "Off · use the date and time of import") {
            work("") { engine!!.state=engine!!.state.copy(preferences=engine!!.state.preferences.copy(preserveImportDates=!state.preferences.preserveImportDates)) }
        }
        setting(content,"Delete by date range","Choose a start and end date and time") { deleteDateRange() }
        setting(content,"Completed to-dos","${state.messages.count { it.deleted==null && it.completedArchived() }} archived · completed items move here after 24 hours") { navigate("completed") }
        setting(content,"Archived starred messages","${state.messages.count { it.deleted==null && it.starArchived() }} messages · restore to Starred") { navigate("archived-stars") }
        setting(content,"To-do widget","Scrollable checklist on the home screen. Locked workspaces hide widget contents.") { requestTodoWidget() }
        setting(content,"Voice transcription",if(VoiceTranscriber.installed(this))"English offline model installed · transcribe from message options" else "Optional English download · 40 MB · runs only when requested") { speechSettings() }
        setting(content,"Recently deleted","${state.messages.count { it.deleted!=null }} messages · recover for 90 days") { navigate("deleted") }
        setting(content,"Categories","Add, rename, pin, or remove category chats") { navigate("categories") }
        setting(content,"Model & learning",if(encoder!=null)"Local learner + MiniLM · active on this phone" else "Built-in local learner · active. MiniLM is optional.") { navigate("model") }
        setting(content,"Reclassify all","Refresh unapproved text using the current learner") { reclassifyAll() }
        setting(content,"Share confirmed examples","JSONL · approved, active text messages only") { exportLabels() }
        content.addView(label("A personal inbox, not a messaging service. No account, server, analytics, or cloud upload. Android's share sheet sends only what you choose to share.",13,muted).apply { setPadding(dp(4),dp(16),dp(4),0) })
    }
    private fun reclassifyAll() {
        val count=shown?.messages?.count { it.visible() && !it.approved && it.trainable() } ?: 0
        if(count==0) { toast("No unapproved text to reclassify"); return }
        dialog("Reclassify $count messages?").setMessage("Refresh unapproved text and reviewed voice transcripts, including manual tags. Approved messages and media without transcripts stay unchanged.")
            .setNegativeButton("Cancel",null).setPositiveButton("Reclassify") { _,_->
                work("Suggestions refreshed") { configure(); engine!!.reclassifyPending { classifier.classify(it) } }
            }.showTracked()
    }
    private fun modelScreen(state:ChatState) {
        toolbar("Model & learning","On this phone",{ navigate("settings") })
        val scroll=ScrollView(this); val content=column().apply { setPadding(dp(20),dp(12),dp(20),dp(24)) }; scroll.addView(content); root.addView(scroll,LinearLayout.LayoutParams(-1,0,1f))
        content.addView(label("Local learner · installed",21,ink,true))
        content.addView(label("Learns from thumbs-up with no download or account. Approved messages are remembered immediately. A small classifier head trains after enough examples.",15,muted).apply { setPadding(0,dp(8),0,dp(16)) })
        modelStatus=label("${if(busy)"Working…" else "Ready"} · ${ChatEngine(state).confirmed().size} distinct approved messages",14,green).also { content.addView(it) }
        content.addView(label("How learning works",16,ink,true).apply { setPadding(0,dp(22),0,dp(8)) })
        content.addView(ModelFlowView(this),LinearLayout.LayoutParams(-1,dp(112)))
        content.addView(label("Thumbs-up stores your label. Thumbs-down only tries another tag. Images are manual. Voice transcripts can be classified.",14,muted))
        val counts=state.categories.map { c->c.title to (ChatEngine(state).confirmed().count { it.category==c.id }) }
        counts.forEach { (name,count)-> val line=row().apply { setPadding(0,dp(10),0,dp(10)) }; line.addView(label(name,15,ink),LinearLayout.LayoutParams(0,-2,1f)); line.addView(label("$count approved${if(count>=6)" ✓" else ""}",13,muted)); content.addView(line) }
        state.report?.let { report->content.addView(label("Latest evaluation: ${report.heldOut} held-out messages. Macro F1 ${"%.2f".format(report.candidate)}; starter ${"%.2f".format(report.baseline)}. ${if(report.adopted)"Personal head active." else "Starter retained."}",14,muted).apply { setPadding(0,dp(14),0,dp(14)) }) }
        content.addView(action("Evaluate learning",{ work("") { runOnUiThread { modelStatus?.text="Training and evaluating on this phone…" }; val result=classifier.train(); engine!!.state=engine!!.state.copy(report=result,model=classifier.learned(),splits=classifier.splits()) } }))
        setting(content,"MiniLM sentence encoder",if(encoder!=null)"Installed · offline · INT8 · ${MiniLm.modelFile(this).length()/1024/1024} MB" else "Optional download · about 23 MB. Adds sentence similarity; the built-in learner already works.") {
            if(encoder==null)dialog("Download MiniLM?").setMessage("Downloads public weights. Your notes stay on your phone.").setNegativeButton("Cancel",null).setPositiveButton("Download") { _,_->work("") {
                MiniLm.download(this) { percent->runOnUiThread { modelStatus?.text="Downloading MiniLM · $percent%" } }
                encoder=MiniLm(this); modelFailure=null; classifier=TextClassifier(encoder); engine!!.state=engine!!.state.copy(model=null,report=null); configure()
            } }.showTracked() else menu("MiniLM",listOf("Remove downloaded encoder")) { work("") { encoder?.close(); encoder=null; MiniLm.modelFile(this).delete(); classifier=TextClassifier(null); engine!!.state=engine!!.state.copy(model=null,report=null); configure() } }
        }
        modelFailure?.let { content.addView(label("MiniLM could not start: $it. The built-in learner is active.",14,muted)) }
        content.addView(label("Encoder weights stay frozen. The personal head uses approved text; held-out labels stay outside training. Learning never runs on dislikes or predictions.",13,muted))
    }
    private fun categories(state:ChatState) {
        toolbar("Categories","One chat per category",{ navigate("settings") },listOf("add" to { editCategory(null) }))
        val scroll=ScrollView(this); val content=column().apply { setPadding(dp(16),dp(10),dp(16),dp(20)) }; scroll.addView(content); root.addView(scroll,LinearLayout.LayoutParams(-1,0,1f))
        state.categories.forEach { c -> setting(content,c.title + if(c.pinned)" · pinned" else "",c.description) { categoryMenu(c) } }
        content.addView(action("Add category",{ editCategory(null) }))
    }
    private fun categoryMenu(c:Category) { menu(c.title,listOf("Edit category",if(c.pinned)"Unpin chat" else "Pin chat","Remove category","Export category")) { i -> when(i) {
        0->editCategory(c); 1->work { engine!!.state=engine!!.state.copy(categories=engine!!.state.categories.map { if(it.id==c.id)it.copy(pinned=!it.pinned) else it }) }
        2->{ val others=shown!!.categories.filterNot { it.id==c.id }; menu("Move its messages before removing",listOf("Unsorted")+others.map { it.title }) { target ->work("Category removed · messages kept") { engine!!.removeCategory(c.id,if(target==0)null else others[target-1].id); configure(); runOnUiThread { if(route==c.id)route="home" } } } }
        3->{ navigate(c.id); exportMenu() }
    } } }
    private fun editCategory(c:Category?) {
        val box=column().apply { setPadding(dp(20),dp(4),dp(20),0) }
        val title=EditText(this).apply { hint="Name"; setText(c?.title.orEmpty()) }; val description=EditText(this).apply { hint="What belongs here?"; setText(c?.description.orEmpty()); maxLines=5 }
        val examples=EditText(this).apply { hint="Example messages (one per line)"; setText(c?.examples?.joinToString("\n").orEmpty()); maxLines=5 }
        box.addView(title); box.addView(description); box.addView(examples)
        dialog(if(c==null)"Add category" else "Edit category").setView(box).setNegativeButton("Cancel",null).setPositiveButton("Save") { _,_->
            val category=Category(c?.id ?: UUID.randomUUID().toString(),title.text.toString().trim(),description.text.toString().trim(),examples.text.toString().lines().map { it.trim() }.filter { it.isNotBlank() }.take(30),c?.pinned ?: false)
            work { if(c==null)engine!!.addCategory(category) else {
                require(category.title.isNotBlank() && category.title.length<=60 && category.description.isNotBlank() && category.description.length<=2000 && category.examples.all { it.length<=2000 })
                require(engine!!.state.categories.none { it.id!=category.id && it.title.equals(category.title,true) }) { "That category name is already used" }
                engine!!.state=engine!!.state.copy(categories=engine!!.state.categories.map { if(it.id==category.id)category else it },model=null,report=null)
            }; configure() }
        }.showTracked()
    }
    private fun deleted(state:ChatState) {
        toolbar("Recently deleted","Recover within 90 days",{ navigate("settings") })
        val rows=state.messages.filter { it.deleted!=null && System.currentTimeMillis()-it.deleted<RETENTION_MS }.sortedByDescending { it.deleted }
        val recycler=recycler().apply { setBackgroundColor(paper) }
        recycler.adapter=object:RecyclerView.Adapter<RowHolder>() {
            override fun getItemCount()=rows.size
            override fun onCreateViewHolder(parent:ViewGroup,type:Int)=RowHolder(column())
            override fun onBindViewHolder(holder:RowHolder,position:Int) {
                holder.box.removeAllViews()
                holder.box.addView(messageBubble(state,rows[position],recovering=true)); holder.box.tag=rows[position].id
            }
        }
        list=recycler; root.addView(recycler,LinearLayout.LayoutParams(-1,0,1f))
        if(rows.isEmpty()) root.addView(label("No deleted messages",15,muted).apply { gravity=Gravity.CENTER; setPadding(0,dp(20),0,dp(20)) })
    }
    private fun restoreMessage(message:Message) {
        if(playingId==message.id)stopPlayback()
        work("Message restored",onSuccess={ if(imageMessageId==message.id)imageDialog?.dismiss() }) { engine!!.restore(message.id); configure() }
    }
    private fun attachMenu() { menu("Add to chat",listOf("Photo / image","File","Voice message","Import TXT / Markdown")) { when(it) { 0->pickImage(); 1->startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply { type="*/*"; addCategory(Intent.CATEGORY_OPENABLE) },10); 2->if(recordVoice()) { recordingLocked=true; updateRecordingComposer() }; 3->pickText() } } }
    private fun pickImage() { startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply { type="image/*"; addCategory(Intent.CATEGORY_OPENABLE) },10) }
    private fun pickText() { startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply { type="*/*"; putExtra(Intent.EXTRA_MIME_TYPES,arrayOf("text/plain","text/markdown","text/x-markdown","application/octet-stream","application/x-markdown")); addCategory(Intent.CATEGORY_OPENABLE) },12) }
    private fun importText(uri:Uri) {
        if(busy || shown==null) { pendingTextUri=uri; return }
        worker.execute { try {
            val source=contentResolver.openInputStream(uri)?.use { input-> val output=java.io.ByteArrayOutputStream(); val buffer=ByteArray(8192); while(true) { val count=input.read(buffer); if(count<0)break; require(output.size()+count<=10*1024*1024) { "Text imports are limited to 10 MB" }; output.write(buffer,0,count) }; output.toString("UTF-8") } ?: error("Cannot read this file")
            val dayFirst=Locale.getDefault().country !in listOf("US","PH")
            runOnUiThread { if(resumed && shown!=null)showImport(source,dayFirst) }
        } catch(error:Exception) { runOnUiThread { tell(error.message ?: "Import failed") } } }
    }
    private fun showImport(source:String,dayFirst:Boolean) {
        val parsed=ChatImport.parse(source,dayFirst)
        if(parsed.entries.isEmpty()) { tell("No text messages found"); return }
        if(parsed.entries.size>5000) { tell("Import at most 5,000 messages at once"); return }
        val box=column().apply { setPadding(dp(20),dp(10),dp(20),dp(8)) }
        box.addView(label("${parsed.entries.size} messages will be added to Inbox and tagged locally. Imported labels are never treated as approvals.",15,ink))
        box.addView(label(if(shown!!.preferences.preserveImportDates)"Transcript dates will be kept." else "Messages use the import date and time. Change this in Settings to keep transcript dates.",13,muted))
        val format=CheckBox(this).apply { text="Dates use day / month / year"; isChecked=dayFirst; visibility=if(parsed.transcript)View.VISIBLE else View.GONE }
        box.addView(format)
        dialog("Import ${if(parsed.transcript)"chat transcript" else "notes"}?").setView(box).setNegativeButton("Cancel",null).setPositiveButton("Import") { _,_->
            val entries=ChatImport.parse(source,format.isChecked).entries
            navigate("all")
            val importedAt=System.currentTimeMillis()
            work("") {
                val incoming=ChatImport.messages(entries,engine!!.state.preferences.preserveImportDates,importedAt).map { message ->
                    classifier.classify(message).let { message.copy(category=it.category,reason=it.reason) } }
                val combined=engine!!.state.messages+incoming
                engine!!.state=engine!!.state.copy(messages=if(engine!!.state.preferences.preserveImportDates)combined.sortedBy { it.created } else combined)
                pendingJump=incoming.firstOrNull()?.id
            }
        }.showTracked()
    }
    private fun importUri(uri:Uri) {
        if(busy || shown==null || recorder!=null) { toast("Finish the current action before adding an attachment"); return }
        var name="attachment"
        contentResolver.query(uri,arrayOf(OpenableColumns.DISPLAY_NAME),null,null,null)?.use { if(it.moveToFirst())name=it.getString(0) ?: name }
        val mime=contentResolver.getType(uri) ?: "application/octet-stream"
        val category=shown?.categories?.firstOrNull { it.id==route }?.id
        val caption=editor?.text?.toString() ?: draft
        val replyId=reply; draft=""; editor?.setText(""); reply=null
        work(if(category==null)"Attachment saved · choose a category" else "",onFailure={ draft=caption; reply=replyId; render() }) {
            val mediaDuration=if(mime.startsWith("audio/"))try { android.media.MediaMetadataRetriever().let { reader->try { reader.setDataSource(this,uri); reader.extractMetadata(android.media.MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L } finally { reader.release() } } } catch(_:Exception) { 0L } else 0L
            val attachment=contentResolver.openInputStream(uri)?.use { vault.importMedia(it,name,mime,mediaDuration) } ?: error("Cannot read attachment")
            engine!!.state=engine!!.state.copy(messages=engine!!.state.messages+Message(text=caption,attachment=attachment,replyTo=replyId,category=category,manual=category!=null,reason="Manually tag this attachment."))
        }
    }
    private fun recordVoice():Boolean {
        if(recorder!=null)return true
        if(busy || shown==null || splitTarget!=null) return false
        if(checkSelfPermission(Manifest.permission.RECORD_AUDIO)!=PackageManager.PERMISSION_GRANTED) { requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO),20); return false }
        stopPlayback()
        try {
            val file=File(cacheDir,"record-${UUID.randomUUID()}.m4a")
            @Suppress("DEPRECATION") val capture=MediaRecorder()
            recorder=capture; recordingFile=file; recordingLocked=false; recordingPaused=false; pausedDuration=0; pausedAt=0; amplitudes.clear()
            capture.setAudioSource(MediaRecorder.AudioSource.MIC); capture.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4); capture.setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
            capture.setAudioEncodingBitRate(64000); capture.setAudioSamplingRate(44100); capture.setOutputFile(file.absolutePath); capture.setMaxDuration(10*60*1000)
            capture.setOnInfoListener { _,what,_->if(what==MediaRecorder.MEDIA_RECORDER_INFO_MAX_DURATION_REACHED)stopRecording(true) }
            capture.prepare(); capture.start(); recordingStarted=System.currentTimeMillis()
            (getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager).hideSoftInputFromWindow(root.windowToken,0)
            if(composerContent!=null)updateRecordingComposer() else { route="all"; render() }
            handler.removeCallbacks(mediaTick); handler.post(mediaTick)
            return true
        } catch(error:Exception) { recorder?.release(); recorder=null; recordingFile?.delete(); recordingFile=null; tell(error.message ?: "Could not start recording"); return false }
    }
    private fun recordingElapsed()=(if(recordingPaused)pausedAt else System.currentTimeMillis())-recordingStarted-pausedDuration
    private fun pauseRecording() {
        try {
            if(recordingPaused) { recorder?.resume(); pausedDuration+=System.currentTimeMillis()-pausedAt; recordingPaused=false }
            else { recorder?.pause(); pausedAt=System.currentTimeMillis(); recordingPaused=true }
            updateRecordingComposer()
        } catch(error:Exception) { tell(error.message ?: "Cannot pause this recording") }
    }
    private fun stopRecording(send:Boolean) {
        if(recorder==null)return
        val file=recordingFile; val elapsed=recordingElapsed()
        handler.removeCallbacks(mediaTick)
        try { recorder?.stop() } catch(_:Exception) { recorder?.release(); recorder=null; file?.delete(); recordingFile=null; if(resumed)render(); return }
        recorder?.release(); recorder=null; recordingFile=null; recordingPaused=false; recordingLocked=false
        if(send && file!=null && elapsed>500) {
            val wave=amplitudes.toList().let { samples->(0 until minOf(32,samples.size)).map { i->samples.drop(i*samples.size/32).take((samples.size/32).coerceAtLeast(1)).maxOrNull() ?: 0 } }
            val category=shown?.categories?.firstOrNull { it.id==route }?.id
            val replyId=reply; reply=null
            work("") { try { val attachment=file.inputStream().use { vault.importMedia(it,"Voice message.m4a","audio/mp4",elapsed) }.copy(waveform=wave); engine!!.state=engine!!.state.copy(messages=engine!!.state.messages+Message(attachment=attachment,replyTo=replyId,category=category,manual=category!=null,reason="Voice messages are manually tagged.")) } finally { file.delete() } }
        } else { file?.delete(); if(resumed)render() }
    }
    private fun openAttachment(m:Message) { if(m.attachment?.mime?.startsWith("audio/")==true)playVoice(m) else if(m.attachment?.mime?.startsWith("image/")==true)showImage(m) else shareAttachment(m,true) }
    private fun playVoice(m:Message,startAt:Int=0) {
        val a=m.attachment ?: return
        if(playingId==m.id && player!=null && !playbackLoading) {
            try { if(player!!.isPlaying)player!!.pause() else if(acquireAudioFocus())player!!.start(); refreshVoiceViews() } catch(_:Exception) { stopPlayback() }
            return
        }
        if(playingId==m.id && playbackLoading)return
        stopPlayback(); playingId=m.id; playbackLoading=true; pendingSeek=startAt; refreshVoiceViews()
        val generation=sessionGeneration
        worker.execute { var file:File?=null
            try {
                file=File(cacheDir,"play-${a.file}.m4a"); vault.readMedia(a).use { input->file.outputStream().use { input.copyTo(it) } }
                val ready=file
                runOnUiThread {
                    if(!resumed || generation!=sessionGeneration || playingId!=m.id) { ready.delete(); return@runOnUiThread }
                    try {
                        playbackFile=ready
                        player=MediaPlayer().apply {
                            setAudioAttributes(audioAttributes); setDataSource(ready.absolutePath)
                            setOnPreparedListener { active->if(playingId==m.id && resumed) { playbackLoading=false; voiceDurations[m.id]=active.duration; if(pendingSeek>0)active.seekTo(pendingSeek); if(acquireAudioFocus()) { active.playbackParams=active.playbackParams.setSpeed(playbackSpeed); active.start() }; handler.removeCallbacks(mediaTick); handler.post(mediaTick); refreshVoiceViews() } else stopPlayback() }
                            setOnCompletionListener { stopPlayback() }
                            setOnErrorListener { _,_,_->stopPlayback(); true }
                            prepareAsync()
                        }
                    } catch(error:Exception) { stopPlayback(); tell(error.message ?: "Cannot play this recording") }
                }
            } catch(error:Exception) { file?.delete(); runOnUiThread { stopPlayback(); tell(error.message ?: "Cannot read this recording") } }
        }
    }
    private fun seekVoice(m:Message,position:Int) {
        if(playingId==m.id) { pendingSeek=position; if(!playbackLoading)try { player?.seekTo(position) } catch(_:Exception) { } }
        else playVoice(m,position)
    }
    private fun changeSpeed() { playbackSpeed=when(playbackSpeed) { 1f->1.5f; 1.5f->2f; else->1f }; if(!playbackLoading)try { player?.let { val wasPlaying=it.isPlaying; it.playbackParams=it.playbackParams.setSpeed(playbackSpeed); if(!wasPlaying)it.pause() } } catch(_:Exception) { }; refreshVoiceViews() }
    private fun updateVoice(id:String,view:VoiceMessageView) {
        val a=shown?.messages?.firstOrNull { it.id==id }?.attachment
        val active=id==playingId && player!=null && !playbackLoading
        try { view.update(active && player!!.isPlaying,if(active)player!!.currentPosition else 0,if(active)player!!.duration else (voiceDurations[id] ?: (a?.duration ?: 0).toInt()),playbackSpeed,id==playingId && playbackLoading) } catch(_:Exception) { view.update(false,0,(voiceDurations[id] ?: (a?.duration ?: 0).toInt()),playbackSpeed) }
    }
    private fun refreshVoiceViews() { voiceViews.forEach { (id,view)->updateVoice(id,view) } }
    private fun stopPlayback() { handler.removeCallbacks(mediaTick); try { player?.stop() } catch(_:Exception) { }; player?.release(); player=null; getSystemService(android.media.AudioManager::class.java).abandonAudioFocusRequest(audioFocus); playbackFile?.delete(); playbackFile=null; playingId=null; playbackLoading=false; refreshVoiceViews() }
    private fun imageBubble(body:LinearLayout,message:Message,width:Int,recovering:Boolean=false) {
        val preview=ImageView(this).apply { scaleType=ImageView.ScaleType.FIT_CENTER; adjustViewBounds=true; maxHeight=dp(340); minimumHeight=dp(120); contentDescription="Image: ${message.attachment?.name}"; tag=message.id; clipToOutline=true; background=rounded(colors.quote,6) }
        body.addView(preview,LinearLayout.LayoutParams(width,-2))
        loadImage(message,preview)
        preview.setOnClickListener { if(selected.isNotEmpty())toggleSelection(message.id) else showImage(message) }
        if(!recovering)preview.setOnLongClickListener { toggleSelection(message.id); true }
    }
    private fun loadImage(message:Message,view:ImageView) {
        val a=message.attachment ?: return; val generation=sessionGeneration
        imageCache.get(a.file)?.let { view.setImageBitmap(it); return }
        worker.execute { try {
            val bytes=vault.readMedia(a).use { it.readBytes() }
            val bounds=BitmapFactory.Options().apply { inJustDecodeBounds=true }; BitmapFactory.decodeByteArray(bytes,0,bytes.size,bounds)
            var sample=1; while(maxOf(bounds.outWidth,bounds.outHeight)/sample>1400)sample*=2
            val decoded=BitmapFactory.decodeByteArray(bytes,0,bytes.size,BitmapFactory.Options().apply { inSampleSize=sample }) ?: error("Unsupported image")
            val orientation=try { androidx.exifinterface.media.ExifInterface(java.io.ByteArrayInputStream(bytes)).getAttributeInt(androidx.exifinterface.media.ExifInterface.TAG_ORIENTATION,1) } catch(_:Exception) { 1 }
            val matrix=android.graphics.Matrix().apply { when(orientation) { 2->setScale(-1f,1f); 3->setRotate(180f); 4->setScale(1f,-1f); 5->{ setRotate(90f); postScale(-1f,1f) }; 6->setRotate(90f); 7->{ setRotate(270f); postScale(-1f,1f) }; 8->setRotate(270f) } }
            val bitmap=if(matrix.isIdentity)decoded else android.graphics.Bitmap.createBitmap(decoded,0,0,decoded.width,decoded.height,matrix,true).also { if(it!==decoded)decoded.recycle() }
            runOnUiThread { if(generation==sessionGeneration && resumed && view.tag==message.id && shown?.messages?.any { it.id==message.id && (it.deleted==null || System.currentTimeMillis()-it.deleted<RETENTION_MS) }==true){ imageCache.put(a.file,bitmap); view.setImageBitmap(bitmap) } else bitmap.recycle() }
        } catch(_:Exception) { runOnUiThread { if(view.tag==message.id) { view.setImageDrawable(ChatIcon("camera",muted)); view.contentDescription="Could not display ${a.name}" } } } }
    }
    private fun showImage(message:Message) {
        val dialog=Dialog(this,android.R.style.Theme_Material_NoActionBar_Fullscreen)
        val content=column().apply { setBackgroundColor(Color.BLACK) }
        imageDialog=dialog; imageMessageId=message.id
        val bar=row(); bar.addView(icon("back","Close image") { dialog.dismiss() }); bar.addView(label(message.attachment?.name.orEmpty(),16,Color.WHITE),LinearLayout.LayoutParams(0,-2,1f)); bar.addView(icon("share","Share image") { shareAttachment(message) }); if(message.deleted==null)bar.addView(icon("delete","Delete image") { confirmDelete(setOf(message.id)) }) else bar.addView(action("Restore",{ restoreMessage(message) },false)); content.addView(bar)
        val image=ZoomImage(this).apply { tag=message.id; contentDescription=message.attachment?.name; scaleType=ImageView.ScaleType.FIT_CENTER }
        content.addView(image,LinearLayout.LayoutParams(-1,0,1f)); dialog.setContentView(content); trackDialog(dialog); dialog.show(); loadImage(message,image)
        if(vault.lockedStorage)dialog.window?.addFlags(WindowManager.LayoutParams.FLAG_SECURE)
    }
    private fun suggest(message: Message) {
        val suggestion=classifier.classify(message)
        engine!!.update(message.id) { it.copy(category=suggestion.category,reason=suggestion.reason) }
    }
    private fun startSplit(message: Message) {
        if(busy || message.attachment!=null)return
        splitPreviousDraft=editor?.text?.toString() ?: draft
        splitPreviousReply=reply
        route="all"; splitTarget=message.id; draft=message.text; reply=null; selected.clear()
        pendingJump=message.id; render(); editor?.requestFocus()
        (getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager).showSoftInput(editor,InputMethodManager.SHOW_IMPLICIT)
    }
    private fun cancelSplit(refresh: Boolean=true) {
        if(splitTarget==null)return
        splitTarget=null; draft=splitPreviousDraft; reply=splitPreviousReply
        editor?.setText(draft)
        if(refresh && shown!=null)render()
    }
    private fun splitControls(box: LinearLayout) {
        val heading=row()
        heading.addView(label("Split message",14,green,true),LinearLayout.LayoutParams(0,-2,1f))
        heading.addView(action("Finish",{ sendSplitPart(Int.MAX_VALUE) },false))
        heading.addView(icon("close","Close split editor") { cancelSplit() }); box.addView(heading)
        val scroll=ScrollView(this)
        val lines=column()
        splitLines=lines; updateSplitLines(draft)
        scroll.addView(lines); box.addView(scroll,LinearLayout.LayoutParams(-1,dp(96)))
        box.addView(label("Send a line, or place the cursor at a split point and tap send. Finish sends the rest.",12,muted))
    }
    private fun updateSplitLines(text:String) {
        val lines=splitLines ?: return
        lines.removeAllViews()
        var offset=0
        text.split('\n').forEachIndexed { index,lineText ->
            offset+=lineText.length+1
            val end=offset.coerceAtMost(text.length)
            if(lineText.isNotBlank()) {
                val line=row(); line.addView(label(lineText.take(100),13,ink).apply { maxLines=1; ellipsize=android.text.TextUtils.TruncateAt.END },LinearLayout.LayoutParams(0,-2,1f))
                line.addView(icon("send","Send through line ${index+1}") { sendSplitPart(end) }); lines.addView(line)
            }
        }
    }
    private fun sendSplitPart(endOffset: Int?=null) {
        if(busy)return
        val target=splitTarget ?: return
        val text=editor?.text?.toString() ?: draft
        if(text.isBlank()) { tell("Keep at least one line, or close the split editor."); return }
        val cursor=editor?.selectionEnd ?: 0
        val end=(endOffset ?: if(cursor in 1 until text.length)cursor else (text.indexOf('\n').takeIf { it>=0 }?.plus(1) ?: text.length)).coerceAtMost(text.length)
        val part=text.substring(0,end).trim()
        val remaining=text.substring(end).trimStart()
        if(part.isBlank())return
        draft=remaining; editor?.setText(remaining)
        var nextTarget:String?=target
        var sentId:String?=null
        work("",onFailure={ draft=text; splitTarget=target; render() },onSuccess={
            splitTarget=nextTarget; pendingJump=sentId
            if(nextTarget==null) { draft=splitPreviousDraft; reply=splitPreviousReply }
        }) {
            val parts=engine!!.split(target,if(remaining.isBlank())listOf(part) else listOf(part,remaining))
            configure(); suggest(parts.first())
            // The remainder is already stored in place, so leaving the editor cannot lose text.
            nextTarget=parts.getOrNull(1)?.id
            sentId=parts.last().id
        }
    }
    private fun toggleTask(message:Message) {
        if(selected.isNotEmpty())toggleSelection(message.id)
        else work("") { val current=engine!!.state.messages.first { it.id==message.id }; engine!!.complete(current.id,!current.done) }
    }
    private fun todoList(state: ChatState) {
        val current=messages(state,false)
        selected.retainAll(current.map { it.id }.toSet())
        if(selected.isNotEmpty())selectionToolbar(state) else toolbar("To-do","",{ navigate("home") },listOf("search" to { beginSearch() },"more" to { chatMenu() }))
        val recycler=recycler()
        recycler.adapter=object:RecyclerView.Adapter<RowHolder>() {
            override fun getItemCount()=current.size
            override fun onCreateViewHolder(parent:ViewGroup,type:Int)=RowHolder(column())
            override fun onBindViewHolder(holder:RowHolder,position:Int) {
                val m=current[position]; holder.box.removeAllViews()
                holder.box.setBackgroundColor(if(m.id in selected)colors.selected else colors.surface)
                val line=TodoRow(this@MainActivity,m,colors,markdown,{ toggleTask(m) },{ toggleSelection(m.id) })
                line.text.linkMenu=::linkActions
                line.footer.addView(label(if(m.done)"Completed · archives after 24 hours" else time(m.created),11,muted),LinearLayout.LayoutParams(0,-2,1f))
                if(m.trainable() && selected.isEmpty())line.footer.addView(action(if(m.approved)"👍 ✓" else "👍",{ work("") { engine!!.approve(m.id); configure(); maybeTrain() } },false))
                holder.box.addView(line); holder.box.tag=m.id
                holder.box.addView(View(this@MainActivity).apply { setBackgroundColor(colors.divider) },LinearLayout.LayoutParams(-1,dp(1)))
            }
        }
        list=recycler; viewport=ChatViewport(this,recycler,current.map { it.id },colors); root.addView(viewport,LinearLayout.LayoutParams(-1,0,1f))
        if(selected.isEmpty())composer(state)
    }
    private fun archived(state: ChatState) {
        val completed=route=="completed"
        toolbar(if(completed)"Completed to-dos" else "Archived starred messages","",{ navigate("settings") })
        val rows=state.messages.filter { it.deleted==null && if(completed)it.completedArchived() else it.starArchived() }
        val recycler=recycler(); recycler.adapter=object:RecyclerView.Adapter<RowHolder>() {
            override fun getItemCount()=rows.size
            override fun onCreateViewHolder(parent:ViewGroup,type:Int)=RowHolder(column())
            override fun onBindViewHolder(holder:RowHolder,position:Int) {
                val m=rows[position]; holder.box.tag=m.id; holder.box.removeAllViews(); holder.box.setPadding(dp(16),dp(12),dp(16),dp(12))
                holder.box.addView(label(m.text.ifBlank { m.attachment?.name.orEmpty() },16,ink))
                val actions=row()
                actions.addView(action("Restore",{ work("") { engine!!.update(m.id) { it.copy(done=false,completedAt=null,starArchiveAt=null) } } },false))
                actions.addView(action("Delete",{ confirmDelete(setOf(m.id)) },false)); holder.box.addView(actions)
            }
        }; list=recycler; root.addView(recycler,LinearLayout.LayoutParams(-1,0,1f))
        if(rows.isEmpty())root.addView(label("No archived messages",15,muted).apply { gravity=Gravity.CENTER; setPadding(0,dp(20),0,dp(20)) })
    }
    private fun deleteDateRange() {
        val from=java.util.Calendar.getInstance().apply { set(java.util.Calendar.HOUR_OF_DAY,0); set(java.util.Calendar.MINUTE,0); set(java.util.Calendar.SECOND,0); set(java.util.Calendar.MILLISECOND,0) }
        val through=java.util.Calendar.getInstance().apply { set(java.util.Calendar.SECOND,59); set(java.util.Calendar.MILLISECOND,999) }
        val content=column().apply { setPadding(dp(20),dp(8),dp(20),dp(8)) }
        val format=SimpleDateFormat("d MMM yyyy, HH:mm",Locale.getDefault())
        val start=label("",16,ink); val end=label("",16,ink)
        fun chooseDate(value:java.util.Calendar,view:TextView) {
            val picker=android.app.DatePickerDialog(this,{ _,year,month,day ->
                value.set(year,month,day)
                val time=android.app.TimePickerDialog(this,{ _,hour,minute -> value.set(java.util.Calendar.HOUR_OF_DAY,hour); value.set(java.util.Calendar.MINUTE,minute); view.text=format.format(value.time) },value.get(java.util.Calendar.HOUR_OF_DAY),value.get(java.util.Calendar.MINUTE),true)
                trackDialog(time); time.show()
            },value.get(java.util.Calendar.YEAR),value.get(java.util.Calendar.MONTH),value.get(java.util.Calendar.DAY_OF_MONTH))
            trackDialog(picker); picker.show()
        }
        start.text=format.format(from.time); end.text=format.format(through.time)
        content.addView(label("From",13,muted)); content.addView(start.apply { minHeight=dp(52); gravity=Gravity.CENTER_VERTICAL; setOnClickListener { chooseDate(from,this) } })
        content.addView(label("Through",13,muted)); content.addView(end.apply { minHeight=dp(52); gravity=Gravity.CENTER_VERTICAL; setOnClickListener { chooseDate(through,this) } })
        dialog("Delete by date range").setView(content).setNegativeButton("Cancel",null).setPositiveButton("Continue") { _,_->
            if(from.timeInMillis>through.timeInMillis) { tell("End must follow start"); return@setPositiveButton }
            val ids=shown!!.messages.filter { it.deleted==null && it.created in from.timeInMillis..through.timeInMillis }.map { it.id }.toSet()
            if(ids.isEmpty())tell("No messages in this range") else confirmDelete(ids)
        }.showTracked()
    }
    private fun requestTodoWidget() {
        val manager=android.appwidget.AppWidgetManager.getInstance(this)
        if(manager.isRequestPinAppWidgetSupported)manager.requestPinAppWidget(android.content.ComponentName(this,TodoWidget::class.java),null,null)
        else tell("Long-press the Android home screen, choose Widgets, then Note Chat To-do.")
    }
    private fun speechSettings() {
        if(VoiceTranscriber.installed(this))menu("Offline speech · English",listOf("Remove downloaded speech model")) { work("") { VoiceTranscriber.remove(this) } }
        else dialog("Download offline speech model?").setMessage("Vosk English: 40 MB download, about 300 MB memory during transcription. Audio stays on this phone. The model is loaded only for transcription and then released.")
            .setNegativeButton("Cancel",null).setPositiveButton("Download") { _,_->work("Speech model installed") { VoiceTranscriber.download(this) { percent -> runOnUiThread { modelStatus?.text="Downloading speech model · $percent%" } } } }.showTracked()
    }
    private fun transcribe(message: Message) {
        if(!VoiceTranscriber.installed(this)) { speechSettings(); return }
        stopPlayback()
        val generation=sessionGeneration
        work("",false) {
            val attachment=message.attachment ?: error("No recording")
            val file=File(cacheDir,"play-transcribe-${UUID.randomUUID()}.m4a")
            val text=try { vault.readMedia(attachment).use { input -> file.outputStream().use { input.copyTo(it) } }; VoiceTranscriber.transcribe(this,file) { !resumed || generation!=sessionGeneration } } finally { file.delete() }
            runOnUiThread { if(resumed && generation==sessionGeneration) {
                val reviewed=if(message.text.isNotBlank() && !message.transcribed)message.text+"\n\n"+text else text
                val input=EditText(this).apply { setText(reviewed); maxLines=8; setTextColor(ink) }
                dialog("Review transcript").setView(input).setNegativeButton("Cancel",null).setPositiveButton("Save & classify") { _,_->work("") {
                    engine!!.edit(message.id,input.text.toString()); engine!!.update(message.id) { it.copy(transcribed=true) }; configure()
                    suggest(engine!!.state.messages.first { it.id==message.id })
                } }.showTracked()
            } }
        }
    }
    private fun duration(ms:Long)="${ms/60000}:${(ms/1000%60).toString().padStart(2,'0')}"
    private fun maybeTrain() {
        val current=engine ?: return
        val confirmed=current.confirmed()
        if(current.state.categories.isEmpty() || current.state.categories.any { c -> confirmed.count { it.category==c.id }<6 })return
        if(current.state.report!=null && confirmed.size-current.state.report!!.confirmed<10)return
        try {
            val report=classifier.train()
            current.state=current.state.copy(report=report,model=classifier.learned(),splits=classifier.splits())
        } catch(_:IllegalArgumentException) {
            // A category change can leave no held-out example; the approval still saves.
        }
    }
    private fun shareAttachment(m:Message,view:Boolean=false) {
        val a=m.attachment ?: return
        work("",false) {
            val directory=File(cacheDir,"shared").apply { mkdirs() }; val file=File(directory,"${a.file}-${a.name.replace(Regex("[^A-Za-z0-9._-]"),"_")}")
            vault.readMedia(a).use { input->file.outputStream().use { input.copyTo(it) } }
            runOnUiThread { if(resumed) { val uri=FileProvider.getUriForFile(this,"$packageName.files",file); val intent=Intent(if(view)Intent.ACTION_VIEW else Intent.ACTION_SEND).apply {
                type=a.mime; if(view)setDataAndType(uri,a.mime) else putExtra(Intent.EXTRA_STREAM,uri); addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION); clipData=ClipData.newRawUri("Attachment",uri)
            }; try { startActivity(Intent.createChooser(intent,if(view)"Open attachment" else "Share attachment")) } catch(_:Exception) { tell("No app can open this file type") } } }
        }
    }
    private fun exportMenu() { menu("Export chat",listOf("Share Markdown · text","Share TXT · text","Save Markdown · text","Save TXT · text","Share ZIP with attachments")) { choice->exportChat(choice) } }
    private fun shareSelection(rows:List<Message>) { exportChat(4,rows) }
    private fun exportChat(choice:Int,selection:List<Message>?=null) {
        val state=shown ?: return; val selected=selection ?: messages(state,false); val name=if(selection==null)title(state) else "Selected messages"
        val markdown=choice!=1 && choice!=3; val zip=choice==4
        work("Export ready",false) {
            val folder=File(cacheDir,"shared").apply { mkdirs() }
            val file=File(folder,"${UUID.randomUUID()}-${name.replace(Regex("[^A-Za-z0-9_-]"),"_")}.${if(zip)"zip" else if(markdown)"md" else "txt"}")
            val text=ChatExport.render(name,selected,state.categories,markdown)
            try {
                if(zip) ZipOutputStream(file.outputStream()).use { output ->
                    output.putNextEntry(ZipEntry("chat.md")); output.write(text.toByteArray()); output.closeEntry()
                    selected.mapNotNull { it.attachment }.distinctBy { it.file }.forEach { attachment->output.putNextEntry(ZipEntry(ChatExport.mediaName(attachment))); vault.readMedia(attachment).use { it.copyTo(output) }; output.closeEntry() }
                } else file.writeText(text)
            } catch(error:Exception) { file.delete(); throw error }
            runOnUiThread { if(resumed) if(choice==2 || choice==3) {
                pendingExport=file; startActivityForResult(Intent(Intent.ACTION_CREATE_DOCUMENT).apply { type=if(markdown)"text/markdown" else "text/plain"; addCategory(Intent.CATEGORY_OPENABLE); putExtra(Intent.EXTRA_TITLE,"${name}.${if(markdown)"md" else "txt"}") },11)
            } else shareFile(file,if(zip)"application/zip" else if(markdown)"text/markdown" else "text/plain") }
        }
    }
    private fun exportLabels() {
        work("Confirmed examples ready",false) {
            val file=File(File(cacheDir,"shared").apply { mkdirs() },"confirmed-${UUID.randomUUID()}.jsonl")
            file.writeText(engine!!.confirmed().joinToString("\n") { org.json.JSONObject().put("text",it.text).put("label",it.category).toString() }+"\n")
            runOnUiThread { if(resumed)shareFile(file,"application/x-ndjson") }
        }
    }
    private fun shareFile(file:File,mime:String) {
        val uri=FileProvider.getUriForFile(this,"$packageName.files",file)
        startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).apply { type=mime; putExtra(Intent.EXTRA_STREAM,uri); clipData=ClipData.newRawUri("Export",uri); addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION) },"Share chat"))
    }
    @Deprecated("Deprecated in Java") override fun onActivityResult(requestCode:Int,resultCode:Int,data:Intent?) {
        super.onActivityResult(requestCode,resultCode,data)
        if(resultCode!=RESULT_OK) { if(requestCode==11)pendingExport=null; return }
        val uri=data?.data ?: return
        if(requestCode==12) { pendingTextUri=uri; if(shown!=null)processPending() }
        if(requestCode==10) { pendingUri=uri; if(shown!=null)processPending() }
        if(requestCode==11) {
            val file=pendingExport ?: return; pendingExport=null
            worker.execute { try { contentResolver.openOutputStream(uri)?.use { output->file.inputStream().use { it.copyTo(output) } } ?: error("Cannot write chosen file"); runOnUiThread { toast("Chat exported") } } catch(e:Exception) { runOnUiThread { tell(e.message ?: "Export failed") } } }
        }
    }
    override fun onRequestPermissionsResult(requestCode:Int,permissions:Array<out String>,results:IntArray) {
        super.onRequestPermissionsResult(requestCode,permissions,results)
        if(requestCode==20 && results.firstOrNull()==PackageManager.PERMISSION_GRANTED) { if(recordVoice()) { recordingLocked=true; updateRecordingComposer() } } else if(requestCode==20)tell("Microphone access is needed only to record a voice message.")
    }
    @Deprecated("Deprecated in Java") override fun onBackPressed() {
        if(recorder!=null) { stopRecording(false); return }
        if(splitTarget!=null) { cancelSplit(); return }
        if(selected.isNotEmpty()) { selected.clear(); render(); return }
        if(searchMode) { searchMode=false; query=""; render(); return }
        if(route=="model") { navigate("settings"); return }
        if(route!="home") { reply=null; draft=editor?.text?.toString() ?: draft; navigate("home") } else super.onBackPressed()
    }
    private fun searchBox() {
        val search=EditText(this).apply { hint="Search messages or categories"; textSize=15f; setSingleLine(); background=rounded(colors.inset,22); setText(query); setPadding(dp(18),dp(10),dp(18),dp(10)) }
        search.addTextChangedListener(object:TextWatcher { override fun beforeTextChanged(s:CharSequence?,start:Int,count:Int,after:Int)=Unit; override fun onTextChanged(s:CharSequence?,start:Int,before:Int,count:Int) { query=s.toString(); list?.adapter?.notifyDataSetChanged() }; override fun afterTextChanged(s:Editable?)=Unit })
        search.setOnEditorActionListener { _,_,_->(getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager).hideSoftInputFromWindow(search.windowToken,0); true }
        root.addView(search,LinearLayout.LayoutParams(-1,dp(46)).apply { leftMargin=dp(16); rightMargin=dp(16); bottomMargin=dp(8) })
    }
    private fun toolbar(title:String,subtitle:String,back:()->Unit,actions:List<Pair<String,()->Unit>> = emptyList()) {
        val bar=row().apply { setPadding(dp(4),dp(4),dp(4),dp(4)); setBackgroundColor(colors.surface) }
        bar.addView(icon("back","Back",back)); val titles=column(); titles.addView(label(title,19,ink,true)); if(subtitle.isNotBlank())titles.addView(label(subtitle,12,muted)); bar.addView(titles,LinearLayout.LayoutParams(0,-2,1f)); actions.forEach { entry->bar.addView(icon(entry.first,entry.first,entry.second).also { if(entry.first=="more")overflowAnchor=it }) }; root.addView(bar)
    }
    private fun setting(parent:LinearLayout,title:String,description:String,click:()->Unit) {
        val box=column().apply { setPadding(dp(4),dp(16),dp(4),dp(16)) }; box.addView(label(title,17,ink,true)); box.addView(label(description,14,muted).apply { setPadding(0,dp(6),0,0) }); box.setOnClickListener { click() }; parent.addView(box)
        parent.addView(View(this).apply { setBackgroundColor(colors.divider) },LinearLayout.LayoutParams(-1,dp(1)))
    }
    private fun recycler()=RecyclerView(this).apply { layoutManager=LinearLayoutManager(this@MainActivity); itemAnimator=null; setHasFixedSize(false) }
    private class RowHolder(val box:LinearLayout):RecyclerView.ViewHolder(box) { init { box.layoutParams=RecyclerView.LayoutParams(-1,-2) } }
    private fun column()=LinearLayout(this).apply { orientation=LinearLayout.VERTICAL }
    private fun row()=LinearLayout(this).apply { orientation=LinearLayout.HORIZONTAL; gravity=Gravity.CENTER_VERTICAL }
    private fun label(text:String,size:Int,color:Int,bold:Boolean=false)=TextView(this).apply { this.text=text; textSize=size.toFloat(); setTextColor(color); if(bold)setTypeface(null,Typeface.BOLD) }
    private fun pill(text:String,color:Int,background:Int)=label(text,12,color).apply { setPadding(dp(9),dp(5),dp(9),dp(5)); this.background=rounded(background,14) }
    private fun rounded(color:Int,radius:Int)=GradientDrawable().apply { setColor(color); cornerRadius=dp(radius).toFloat() }
    private fun icon(name:String,description:String,click:()->Unit)=ImageButton(this).apply { contentDescription=description; setImageDrawable(ChatIcon(name,muted)); background=null; setPadding(dp(12),dp(12),dp(12),dp(12)); layoutParams=LinearLayout.LayoutParams(dp(48),dp(48)); setOnClickListener { click() } }
    private fun action(title:String,click:()->Unit,solid:Boolean=true)=TextView(this).apply { text=title; textSize=if(solid)15f else 12f; gravity=Gravity.CENTER; setTextColor(if(solid)Color.WHITE else green); setPadding(dp(12),dp(9),dp(12),dp(9)); minimumHeight=dp(40); if(solid)background=rounded(green,20); setOnClickListener { click() }; isClickable=true; contentDescription=title }
    private fun dialog(title:String)=MaterialAlertDialogBuilder(this).setTitle(title)
    private fun trackDialog(dialog:Dialog) { dialogs.add(dialog); dialog.setOnDismissListener { dialogs.remove(dialog); if(imageDialog===dialog) { imageDialog=null; imageMessageId=null } } }
    private fun AlertDialog.Builder.showTracked()=create().also { alert->trackDialog(alert); alert.show(); alert.window?.setWindowAnimations(R.style.DialogMotion); if(vault.lockedStorage)alert.window?.addFlags(WindowManager.LayoutParams.FLAG_SECURE) }
    private fun menu(title:String,options:List<String>,selected:(Int)->Unit) {
        val sheet=BottomSheetDialog(this)
        val scroll=ScrollView(this); val content=column().apply { setPadding(dp(20),dp(12),dp(20),dp(24)) }
        content.addView(label(title,17,ink,true).apply { setPadding(0,dp(12),0,dp(16)); maxLines=2; ellipsize=android.text.TextUtils.TruncateAt.END })
        options.forEachIndexed { index,option->content.addView(label(option,16,ink).apply { minimumHeight=dp(52); gravity=Gravity.CENTER_VERTICAL; setPadding(dp(4),dp(12),dp(4),dp(12)); val value=android.util.TypedValue(); theme.resolveAttribute(android.R.attr.selectableItemBackground,value,true); setBackgroundResource(value.resourceId); setOnClickListener { sheet.dismiss(); selected(index) } }) }
        scroll.addView(content); sheet.setContentView(scroll); trackDialog(sheet); sheet.show(); if(vault.lockedStorage)sheet.window?.addFlags(WindowManager.LayoutParams.FLAG_SECURE)
    }
    private fun overflow(options:List<String>,selected:(Int)->Unit) {
        val anchor=overflowAnchor ?: return menu("Options",options,selected)
        val popup=PopupMenu(this,anchor,Gravity.END)
        options.forEachIndexed { index,option->popup.menu.add(0,index,index,option) }
        activeOverflow?.dismiss(); activeOverflow=popup; popup.setOnDismissListener { if(activeOverflow===popup)activeOverflow=null }; popup.setOnMenuItemClickListener { selected(it.itemId); true }; popup.show()
    }
    private fun tell(message:String) { if(!isFinishing && resumed)dialog("Note Chat").setMessage(message).setPositiveButton("OK",null).showTracked() }
    private fun toast(message:String) { if(resumed)Toast.makeText(this,message,Toast.LENGTH_SHORT).show() }
    private fun dp(value:Int)=(value*resources.displayMetrics.density).toInt()
    private fun time(value:Long)=SimpleDateFormat("HH:mm",Locale.getDefault()).format(Date(value))
    private fun day(value:Long)=SimpleDateFormat("EEE, d MMM",Locale.getDefault()).format(Date(value))
}
