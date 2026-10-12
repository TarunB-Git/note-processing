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
import androidx.core.view.doOnLayout
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
    private val modWorker = Executors.newSingleThreadExecutor()
    private lateinit var vault: Vault
    private var engine: ChatEngine? = null
    private var encoder: MiniLm? = null
    private var classifier = TextClassifier(null,Diagnostics::record)
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
    private var followLatest = false
    private var markdownError: String? = null
    private var pendingFolder: Uri? = null
    private var pendingCategoryPhoto: String? = null
    private var restoredCategoryPhoto: String? = null
    private var pinCursor = 0
    private var headerView: View? = null
    private var editTarget: String? = null
    private var editPreviousDraft = ""
    private var guideProgress=GuideProgress()
    private val drafts=mutableMapOf<String,ComposerDraft>()
    private var homeTab="home"
    private var pendingCaret: Int?=null
    private var forwardAction: (List<Message>)->Unit = ::forwardMessages
    private var guideReveal=false
    private var guideAnchor: View?=null
    private var guideForeground: android.graphics.drawable.Drawable?=null
    private var modBusy = false
    private var queuedMods=0
    private data class PendingChange(val id: Long,val preview: (ChatEngine)->Unit)
    private val pendingChanges=mutableListOf<PendingChange>()
    private var changeId=0L
    private var persistState: (ChatState)->Unit = { vault.save(it) }
    private var exportFolder: (ChatEngine)->Int = { current -> MarkdownFolder(this,current.state.markdown.tree!!).sync(current,vault,::configure,::predict) }
        @Volatile private var markdownDirty=false
    private var lastInteraction=android.os.SystemClock.uptimeMillis()
    private val projections = Runnable {
        if(resumed && shown!=null) {
            if(busy || recorder!=null || selected.isNotEmpty() || dialogs.isNotEmpty() || android.os.SystemClock.uptimeMillis()-lastInteraction<30_000) scheduleIdleSave()
            else {
                val generation=sessionGeneration
                worker.execute {
                    val before=engine?.state
                    if(markdownDirty)flushMarkdown()
                    val state=engine?.state ?: return@execute
                    publishWidget(state); TodoWidget.refresh(this); Reminders.sync(this,state)
                    runOnUiThread { if(resumed && generation==sessionGeneration) {
                        shown=projectPending(state)
                        if(before?.messages!=state.messages)refreshMessages()
                        if(markdownDirty)scheduleIdleSave()
                    } }
                }
            }
        }
    }
    private val markdownTick = object: Runnable {
        override fun run() {
            if(resumed && !busy && android.os.SystemClock.uptimeMillis()-lastInteraction>=30_000 && shown?.markdown?.tree!=null && selected.isEmpty() && recorder==null && dialogs.isEmpty()) {
                val generation=sessionGeneration
                busy=true
                draft=editor?.text?.toString() ?: draft
                worker.execute {
                    val before=engine?.state
                    flushMarkdown(); configure()
                    val after=engine?.state
                    after?.let(::publishWidget)
                    runOnUiThread {
                        busy=false
                        if(generation==sessionGeneration && resumed && after!=null) {
                            shown=projectPending(after)
                            if(before?.messages!=after.messages)refreshMessages()
                        }
                    }
                }
            }
            if(resumed)handler.postDelayed(this,60_000)
        }
    }
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
    private var composerRoot: LinearLayout? = null
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
    private val colors get() = UiColors(isDark(),shown?.preferences?.frontend=="notes")
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
    private var renderedPins=emptySet<String>()
    private fun pinIds(state:ChatState)=state.messages.filter { it.visible() && (it.pinnedUntil ?: 0)>System.currentTimeMillis() }.map { it.id }.toSet()
    private fun archiveIds(state:ChatState)=state.messages.filter { it.deleted==null && (it.completedArchived() || it.starArchived()) }.map { it.id }.toSet()
    private val archiveTick = object : Runnable {
        override fun run() {
            if(resumed && !busy && shown!=null && recorder==null && splitTarget==null && !searchMode && query.isBlank() && selected.isEmpty() && (archiveIds(shown!!)!=renderedArchives || pinIds(shown!!)!=renderedPins)) {
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
        savedInstanceState?.let { pendingCategoryPhoto=it.getString("categoryPhoto"); restoredNavigation=Navigation(it.getString("screen","home"),it.getString("anchor"),it.getInt("index"),it.getInt("offset"),it.getInt("scroll")) }
        captureIntent(intent)
        cacheDir.listFiles()?.filter { file -> listOf("record-","play-","restore-","import-","ocr-").any { file.name.startsWith(it) } }?.forEach { it.deleteRecursively() }
        File(cacheDir,"shared").listFiles()?.filter { System.currentTimeMillis()-it.lastModified()>10*60*1000 }?.forEach { it.delete() }
        if (vault.lockedStorage) lockedScreen() else loadWorkspace()
    }
    override fun onNewIntent(intent: Intent) { super.onNewIntent(intent); setIntent(intent); captureIntent(intent); if (shown != null) processPending() }
    private fun captureIntent(intent: Intent?) {
        intent?.getStringExtra("chat_route")?.let { route=it; intent.removeExtra("chat_route") }
        intent?.getStringExtra("jump_message")?.let { id -> pendingJump=id; highlightJump=true; intent.removeExtra("jump_message") }
        if(intent?.getStringExtra("widget_route")=="todos") {
            route="todos"; pendingWidgetToggle=intent.getStringExtra("widget_complete"); pendingWidgetJump=intent.getStringExtra("widget_jump")
            intent.removeExtra("widget_complete"); intent.removeExtra("widget_jump"); intent.removeExtra("widget_route")
        }
        if (intent?.action != Intent.ACTION_SEND) return
        pendingText = intent.getStringExtra(Intent.EXTRA_TEXT)
        @Suppress("DEPRECATION") pendingUri = intent.getParcelableExtra(Intent.EXTRA_STREAM)
    }
    override fun onUserInteraction() {
        super.onUserInteraction(); lastInteraction=android.os.SystemClock.uptimeMillis()
        if(markdownDirty)scheduleIdleSave()
    }
    private fun scheduleIdleSave() {
        handler.removeCallbacks(projections)
        handler.postDelayed(projections,30_000)
    }
    override fun onResume() {
        super.onResume(); resumed=true
        if(vault.lockedStorage && shown==null && !authInProgress) { lockedScreen(); if(!autoUnlockAttempted) { autoUnlockAttempted=true; unlockWorkspace() } }
        else if(shown!=null && !vault.lockedStorage) {
            val generation=sessionGeneration
            worker.execute { configure(); val state=engine?.state; state?.let(::publishWidget); TodoWidget.refresh(this); runOnUiThread {
                if(state!=null && resumed && generation==sessionGeneration) {
                    draft=editor?.text?.toString() ?: draft; shown=projectPending(state); render(); processPending()
                }
            } }
        }
        handler.removeCallbacks(archiveTick); handler.postDelayed(archiveTick,60_000)
        handler.removeCallbacks(markdownTick); handler.postDelayed(markdownTick,60_000)
        if(markdownDirty)scheduleIdleSave()
    }
    override fun onConfigurationChanged(configuration:Configuration) {
        super.onConfigurationChanged(configuration)
        draft=editor?.text?.toString() ?: draft
        if(shown!=null)render() else lockedScreen()
    }
    override fun onSaveInstanceState(outState: Bundle) {
        val location=capturePosition()
        outState.putString("categoryPhoto",pendingCategoryPhoto)
        outState.putString("screen",location.route); outState.putString("anchor",location.anchor)
        outState.putInt("index",location.index); outState.putInt("offset",location.offset); outState.putInt("scroll",location.scroll)
        super.onSaveInstanceState(outState)
    }
    override fun onStop() {
        super.onStop(); resumed=false; sessionGeneration++; autoUnlockAttempted=false; handler.removeCallbacks(archiveTick); handler.removeCallbacks(markdownTick); handler.removeCallbacks(projections); pendingChanges.clear(); cancelSplit(false); activeOverflow?.dismiss(); activeOverflow=null
        stopPlayback()
        if (recorder != null) stopRecording(send=false)
        val savedDraft=editor?.text?.toString() ?: draft
        val savedReply=reply
        val location=capturePosition(); positions[location.route]=location
        rememberDraft()
        val savedSession=sessionState(shown ?: ChatState(emptyList()))
        if(shown!=null)worker.execute { try { engine?.let { it.state=it.state.copy(draft=savedDraft,replyDraft=savedReply,navigation=location,editDraftId=savedSession.editDraftId,preEditDraft=savedSession.preEditDraft,drafts=savedSession.drafts,locations=savedSession.locations,guide=savedSession.guide,homeTab=savedSession.homeTab); vault.save(it.state); if(markdownDirty)flushMarkdown(); publishWidget(it.state); TodoWidget.refresh(this); Reminders.sync(this,it.state) } } catch(_:Exception) { } }
        if (vault.lockedStorage && !authInProgress) {
            imageCache.evictAll()
            selected.clear(); searchMode=false
            draft=""; reply=null; editTarget=null; editPreviousDraft=""; drafts.clear()
            guideAnchor?.foreground=guideForeground; guideAnchor=null; guideForeground=null
            dialogs.toList().forEach { it.dismiss() }; dialogs.clear()
            shown=null; list=null; viewport=null; editor=null; root.removeAllViews(); lockedScreen()
            worker.execute { vault.close(); engine=null; encoder?.close(); encoder=null; classifier=TextClassifier(null,Diagnostics::record) }
        }
    }
    override fun onDestroy() {
        super.onDestroy(); if(noisyRegistered) { unregisterReceiver(noisyReceiver); noisyRegistered=false }; handler.removeCallbacksAndMessages(null); stopPlayback(); if(recorder!=null) stopRecording(false)
        worker.execute { vault.close(); encoder?.close(); encoder=null }; worker.shutdown(); modWorker.shutdown()
    }
    private fun loadWorkspace() {
        loading("Opening your chats…")
        worker.execute {
            try {
                if (!vault.isOpen) vault.openWithoutLock()
                val restored = ChatEngine(vault.load())
                val expired=restored.expire()
                restored.state=restored.state.copy(messages=restored.state.messages.map { if(it.done && it.completedAt==null)it.copy(completedAt=System.currentTimeMillis()) else it })
                engine=restored
                vault.save(restored.state)
                publishWidget(restored.state)
                TodoWidget.refresh(this)
                val retainedMedia=BackupArchive.attachments(restored.state).map { it.file }.toSet()
                expired.mapNotNull { it.attachment }.distinctBy { it.file }.filterNot { it.file in retainedMedia }.forEach { vault.removeMedia(it) }
                engine=restored
                encoder?.close()
                encoder=try { if(restored.state.preferences.classification && MiniLm.modelFile(this).exists()) MiniLm(this) else null } catch(error:Exception) { modelFailure=error.message; null }
                classifier=TextClassifier(encoder,Diagnostics::record)
                configure()
                markdownDirty=restored.state.markdown.tree!=null
                val state=restored.state
                runOnUiThread { if (!isFinishing && (!vault.lockedStorage || resumed)) { shown=projectPending(state); busy=false; restoreSession(state); privacy(); render(); processPending(); if(markdownDirty)scheduleIdleSave() } }
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
    private fun restoreSession(state: ChatState) {
        drafts.clear(); drafts.putAll(state.drafts)
        positions.putAll(state.locations)
        guideProgress=state.guide
        homeTab=state.homeTab
        draft=state.draft; reply=state.replyDraft; editTarget=state.editDraftId?.takeIf { id->state.messages.any { it.id==id && it.visible() } }; editPreviousDraft=state.preEditDraft
        restoreLocation(state)
        if(isNotebook(route) && route !in drafts)drafts[route]=ComposerDraft(draft,reply,editTarget,editPreviousDraft)
        drafts[route]?.let { pendingCaret=it.cursor.takeIf { cursor->cursor>=0 } }
    }
    private fun restoreLocation(state:ChatState) {
        val location=restoredNavigation ?: if(renderedRoute.isBlank())state.navigation else positions[route]
        restoredNavigation=null
        if(location!=null) {
            if(route=="home" && renderedRoute.isBlank())route=location.route
            positions[location.route]=location
        }
        if(route !in setOf("home","chats","all","starred","unsorted","settings","model","categories","deleted","completed","archived-stars") && !route.startsWith("settings:") && route !in setOf("mods","guide","quiz") && state.categories.none { it.id==route } && state.chats.none { "chat:${it.id}"==route })route="home"
    }
    private fun restorePosition(state:ChatState) {
        val recycler=list
        val manager=recycler?.layoutManager as? LinearLayoutManager
        val location=positions[route]
        if(location!=null && recycler!=null && manager!=null && (recycler.adapter?.itemCount ?: 0)>0) {
            val ids=when(route) {
                "deleted"->state.messages.filter { it.deleted!=null && System.currentTimeMillis()-it.deleted<RETENTION_MS }.sortedByDescending { it.deleted }.map { it.id }
                "completed"->state.messages.filter { it.deleted==null && it.completedArchived() }.map { it.id }
                "archived-stars"->state.messages.filter { it.deleted==null && it.starArchived() }.map { it.id }
                "home", "chats"->emptyList()
                else->messages(state,false).map { it.id }
            }
            val anchor=location.anchor?.let { ids.indexOf(it) } ?: -1
            manager.scrollToPositionWithOffset((if(anchor>=0)anchor else location.index).coerceIn(0,recycler.adapter!!.itemCount-1),location.offset)
        }
        if(followLatest && viewport!=null) { followLatest=false; viewport!!.goLatest() }
        if(location!=null)(0 until root.childCount).map { root.getChildAt(it) }.filterIsInstance<ScrollView>().firstOrNull()?.let { it.post { it.scrollTo(0,location.scroll) } }
        pendingJump?.let { id->
            pendingJump=null
            val highlighted=highlightJump; highlightJump=false
            if(viewport!=null)viewport!!.reveal(id,highlighted)
            else { val index=messages(state,false).indexOfFirst { it.id==id }; if(index>=0)manager?.scrollToPositionWithOffset(index,dp(30)) }
        }
    }
    private fun configure() {
        val current=engine ?: return
        if(!current.state.preferences.classification) {
            if(encoder!=null) { encoder?.close(); encoder=null; classifier=TextClassifier(null,Diagnostics::record) }
            classifier.configure(emptyList(),emptyList())
            return
        }
        if(encoder==null && modelFailure==null && MiniLm.modelFile(this).exists()) {
            try { encoder=MiniLm(this); classifier=TextClassifier(encoder,Diagnostics::record) }
            catch(error:Exception) { modelFailure=error.message ?: "Could not load encoder" }
        }
        classifier.configure(current.state.categories,current.confirmed(),current.state.model,current.state.splits)
    }
    private fun predict(message: Message) = if(engine?.state?.preferences?.classification==false || !message.trainable())Suggestion(null,if(message.chatId!=null)"Private chat · classifier ignored" else "Choose a category manually") else classifier.classify(message)
    private fun projectPending(state: ChatState): ChatState {
        val preview=ChatEngine(state)
        pendingChanges.forEach { pending -> runCatching { pending.preview(preview) } }
        return preview.state
    }
    private fun fast(train: Boolean=false, success: (() -> Unit)?=null, preview: (ChatEngine)->Unit, commit: (ChatEngine)->Unit=preview) {
        val base=shown ?: return
        val optimistic=ChatEngine(base)
        try { preview(optimistic) } catch(error:Exception) { tell(error.message ?: "Cannot change this message"); return }
        rememberDraft()
        val ui=sessionState(base)
        val id=++changeId; val generation=sessionGeneration
        pendingChanges.add(PendingChange(id,preview)); shown=optimistic.state
        refreshMessages()
        worker.execute {
            val current=engine ?: return@execute
            val before=current.state
            try {
                commit(current)
                current.state=Reminders.infer(before,current.state)
                current.state=current.state.copy(draft=ui.draft,replyDraft=ui.replyDraft,editDraftId=ui.editDraftId,preEditDraft=ui.preEditDraft,drafts=ui.drafts,locations=ui.locations,guide=ui.guide,homeTab=ui.homeTab)
                persistState(current.state)
                if(train) {
                    val confirmedState=current.state
                    configure(); maybeTrain()
                    if(current.state!=confirmedState)persistState(current.state)
                }
                Reminders.sync(this,current.state)
                markdownDirty=true
                val state=current.state
                runOnUiThread {
                    pendingChanges.removeAll { it.id==id }
                    if(generation==sessionGeneration && resumed && shown!=null) {
                        shown=projectPending(state); refreshMessages(); success?.invoke()
                        scheduleIdleSave()
                    }
                }
            } catch(error:Exception) {
                current.state=before
                if(train)configure()
                runOnUiThread {
                    pendingChanges.removeAll { it.id==id }
                    if(generation==sessionGeneration && resumed && shown!=null) { shown=projectPending(before); refreshMessages(); tell(error.message ?: "Change was not saved") }
                }
            }
        }
    }
    private fun refreshMessages() {
        val state=shown ?: return
        val adapter=list?.adapter as? MessageRows
        if(adapter==null) { render(); return }
        val current=messages(state,false)
        val wasAtBottom=list?.canScrollVertically(1)!=true
        selected.retainAll(current.map { it.id }.toSet())
        adapter.replace(current); viewport?.updateIds(current.map { it.id })
        if(wasAtBottom || followLatest) { followLatest=false; viewport?.goLatest() }
    }
    private fun approveMessages(ids: Set<String>) {
        val changed=selected.toSet(); selected.clear(); refreshSelection(changed)
        fast(train=true,preview={ current -> ids.forEach { current.approve(it) } })
    }
    private fun disapprove(message: Message) {
        fast(train=true,preview={ current -> current.update(message.id) { it.copy(approved=false,category=null,reason="Trying another category…") } },commit={ current ->
            val actual=current.state.messages.first { it.id==message.id }
            classifier.configure(current.state.categories,current.confirmed().filterNot { it.id==message.id },if(actual.approved)null else current.state.model,current.state.splits)
            current.reject(message.id) { predict(it) }
        })
    }
    private fun publishWidget(state: ChatState) {
        try { WidgetTasks(this).publish(state) } catch(_:Exception) { runOnUiThread { toast("Could not refresh the home-screen checklist") } }
    }
    private fun flushMarkdown() { if(syncMarkdown())markdownDirty=false }
    private fun syncMarkdown(): Boolean {
        val current=engine ?: return false
        if(current.state.markdown.tree==null)return true
        try {
            val conflicts=exportFolder(current)
            markdownError=null
            if(conflicts>0)runOnUiThread { toast("Markdown edits conflicted. Copies were saved beside the category files.") }
            return true
        } catch(error:Exception) {
            val message=error.message ?: "Markdown sync failed"
            if(markdownError!=message)runOnUiThread { toast("Markdown sync: $message") }
            markdownError=message
            return false
        }
    }
    private fun markdownSettings() {
        if(shown?.markdown?.tree==null)chooseMarkdownFolder()
        else menu("Markdown folder",listOf("Sync now","Choose another folder","Disconnect folder")) { choice -> when(choice) {
            0->work("Markdown sync checked") { flushMarkdown() }
            1->chooseMarkdownFolder()
            2->work("Folder disconnected · existing files kept") { engine!!.state=engine!!.state.copy(markdown=MarkdownState()); markdownError=null }
        } }
    }
    private fun chooseMarkdownFolder() {
        dialog("Connect a Markdown folder?").setMessage("Category files and attachments will be readable in a Note Chat subfolder. Edits sync when this app is open or reopened. App lock does not protect those external copies.")
            .setNegativeButton("Cancel",null).setPositiveButton("Choose folder") { _,_->
                startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT_TREE).apply {
                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION or Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION or Intent.FLAG_GRANT_PREFIX_URI_PERMISSION)
                },13)
            }.showTracked()
    }
    private fun openMarkdownLink(target: String) {
        val state=shown ?: return
        val name=target.substringBefore('#').removeSuffix(".md")
        val category=(listOf<String?>(null)+state.categories.map { it.id }).firstOrNull {
            MarkdownWorkspace.filename(state,it).removeSuffix(".md")==name || state.categories.firstOrNull { c->c.id==it }?.title==name
        }
        val known=name=="Unsorted" || category!=null
        val block=target.substringAfter("#^nc-","").takeIf { target.contains("#^nc-") }
        if(block!=null && state.messages.any { it.id==block && it.visible() }) { jumpTo(block,true); return }
        if(known) { navigate(category ?: "unsorted"); return }
        val tree=state.markdown.tree
        if(tree==null) { tell("Connect a Markdown folder to open this local file."); return }
        worker.execute {
            try { val uri=MarkdownFolder(this,tree).link(target) ?: error("Markdown file was not found")
                runOnUiThread { if(resumed)try { startActivity(Intent.createChooser(Intent(Intent.ACTION_VIEW).setDataAndType(uri,"text/markdown").addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION),"Open Markdown file")) } catch(_:Exception) { tell("No Markdown editor is installed") } }
            } catch(error:Exception) { runOnUiThread { tell(error.message ?: "Could not open Markdown file") } }
        }
    }
    private fun exportModelReport() {
        work("Model report ready",false) {
            val file=File(File(cacheDir,"shared").apply { mkdirs() },"model-report-${UUID.randomUUID()}.json")
            file.writeText(ModelReport.encode(engine!!.state,encoder?.identity,android.os.Debug.getPss().coerceAtMost(Int.MAX_VALUE.toLong()).toInt()))
            runOnUiThread { if(resumed)shareFile(file,"application/json") }
        }
    }
    private fun connectFolder(uri: Uri) {
        val state=shown ?: return
        worker.execute {
            try {
                val recovered=MarkdownFolder(this,uri.toString()).recover(state.categories)
                runOnUiThread {
                    if(!resumed || shown==null)return@runOnUiThread
                    val restore={ work("Folder connected · files survive uninstall") {
                        val current=engine!!
                        val saved=MarkdownFolder(this,uri.toString()).recover(current.state.categories,vault)
                        if(saved!=null) {
                            val known=current.state.messages.map { it.id }.toSet()
                            if(current.state.messages.isEmpty())current.state=saved
                            else current.state=current.state.copy(categories=current.state.categories+saved.categories.filter { c->current.state.categories.none { it.id==c.id } },
                                chats=current.state.chats+saved.chats.filter { c->current.state.chats.none { it.id==c.id } },messages=current.state.messages+saved.messages.filterNot { it.id in known })
                            current.state=current.state.copy(markdown=saved.markdown)
                        } else current.state=current.state.copy(markdown=MarkdownState(uri.toString()))
                        configure()
                    } }
                    if(recovered!=null && state.messages.isNotEmpty()) {
                        val duplicates=recovered.messages.count { message->state.messages.any { it.id==message.id } }
                        dialog("Saved workspace found").setMessage("${recovered.messages.size} saved messages; $duplicates already exist here. Existing messages keep their current version. Files remain in the folder until you delete them.")
                            .setNegativeButton("Cancel",null).setPositiveButton("Connect & restore") { _,_->restore() }.showTracked()
                    } else restore()
                }
            } catch(error:Exception) { runOnUiThread { tell(error.message ?: "Cannot restore this folder") } }
        }
    }
    private fun password(title: String, action: (CharArray)->Unit) {
        val input=EditText(this).apply { hint="Backup password · at least 8 characters"; inputType=android.text.InputType.TYPE_CLASS_TEXT or android.text.InputType.TYPE_TEXT_VARIATION_PASSWORD; setPadding(dp(20),dp(12),dp(20),dp(12)) }
        dialog(title).setView(input).setNegativeButton("Cancel",null).setPositiveButton("Continue") { _,_->
            val value=input.text.toString().toCharArray(); input.text.clear()
            if(value.size<8) { value.fill('\u0000'); tell("Use at least eight characters"); return@setPositiveButton }
            action(value)
        }.showTracked()
    }
    private fun createBackup(modelOnly: Boolean=false) {
        if(modelOnly && shown?.model==null) { tell("No personal head has been adopted yet. Train & compare after approving examples, or create a full workspace backup."); return }
        menu(if(modelOnly)"Export personal weights" else "Back up workspace",listOf("Save ZIP · device or Drive","Share ZIP · Drive or another app")) { destination ->
            password(if(modelOnly)"Encrypt model weights" else "Encrypt workspace backup") { secret ->
                work("Backup ready",false) {
                    try {
                        val original=engine!!.state
                        val state=if(modelOnly)ChatState(original.categories.map { it.copy(photo=null) },model=original.model,report=original.report) else original
                        val file=File(File(cacheDir,"shared").apply { mkdirs() },"note-chat-${if(modelOnly)"model" else "backup"}-${System.currentTimeMillis()}.zip")
                        BackupArchive.write(file,state,secret,if(modelOnly)"model" else "workspace") { vault.readMedia(it) }
                        runOnUiThread { if(resumed) {
                            if(destination==0) { pendingExport=file; startActivityForResult(Intent(Intent.ACTION_CREATE_DOCUMENT).apply { type="application/zip"; addCategory(Intent.CATEGORY_OPENABLE); putExtra(Intent.EXTRA_TITLE,file.name) },11) }
                            else shareFile(file,"application/zip")
                        } }
                    } finally { secret.fill('\u0000') }
                }
            }
        }
    }
    private fun isZip(uri: Uri): Boolean {
        val mime=contentResolver.getType(uri).orEmpty()
        val name=contentResolver.query(uri,arrayOf(OpenableColumns.DISPLAY_NAME),null,null,null)?.use { if(it.moveToFirst())it.getString(0) else "" }.orEmpty()
        return mime.contains("zip") || name.endsWith(".zip",true)
    }
    private fun importZip(uri: Uri) {
        worker.execute { try {
            val file=File(cacheDir,"import-${UUID.randomUUID()}.zip")
            contentResolver.openInputStream(uri)?.use { input -> file.outputStream().use { output ->
                val buffer=ByteArray(8192); var total=0L
                while(true) { val n=input.read(buffer); if(n<0)break; total+=n; require(total<=512L*1024*1024) { "ZIP exceeds 512 MB" }; output.write(buffer,0,n) }
            } } ?: error("Cannot read ZIP")
            val kind=BackupArchive.kind(file)
            if(kind!=null)runOnUiThread { if(resumed && shown!=null)password("Unlock ${if(kind=="model")"model" else "workspace"} backup") { secret ->
                worker.execute { try {
                    val restored=try { BackupArchive.read(file,secret,cacheDir) } finally { secret.fill('\u0000'); file.delete() }
                    runOnUiThread { if(resumed && shown!=null)previewRestore(restored) else restored.close() }
                } catch(error:Exception) { file.delete(); runOnUiThread { tell("Could not unlock backup. Check the password and file integrity.") } } }
            } else file.delete() }
            else {
                val candidates=try { ZipTranscript.read(file) } finally { file.delete() }
                runOnUiThread { if(resumed && shown!=null) {
                    if(candidates.isEmpty())tell("ZIP contains no readable TXT or Markdown notes")
                    else if(candidates.size==1)showImport(candidates.first().source,Locale.getDefault().country !in listOf("US","PH"))
                    else menu("Choose transcript from ZIP",candidates.map { "${it.name} · ${it.messages} messages" }) { showImport(candidates[it].source,Locale.getDefault().country !in listOf("US","PH")) }
                } }
            }
        } catch(error:Exception) { runOnUiThread { tell(error.message ?: "ZIP import failed") } } }
    }
    private fun previewRestore(restored: BackupArchive.Restored) {
        val state=shown ?: run { restored.close(); return }
        if(restored.kind=="model") {
            dialog("Import personal weights?").setOnCancelListener { restored.close() }.setMessage("Restores the small head and category definitions. Notes and example memory are excluded. The same sentence encoder is required.")
                .setNegativeButton("Cancel") { _,_->restored.close() }.setPositiveButton("Import") { _,_->work("Personal head imported",onFailure={ restored.close() }) {
                    try {
                        val incoming=restored.state
                        val ids=incoming.categories.map { it.id }.toSet()
                        require(engine!!.state.messages.all { it.category==null || it.category in ids }) { "Model has different categories. Restore its full backup into a new workspace instead." }
                        val identity=if(MiniLm.modelFile(this).exists())MiniLm.CHECKSUM else null
                        require(TextClassifier.compatible(incoming.model!!,incoming.categories,identity)) { "Install the matching MiniLM encoder before importing these weights" }
                        engine!!.state=engine!!.state.copy(categories=incoming.categories,model=incoming.model,report=incoming.report,quiz=null,importedArchives=engine!!.state.importedArchives+restored.fingerprint)
                        configure()
                    } finally { restored.close() }
                } }.showTracked()
            return
        }
        val duplicate=restored.state.messages.count { m->state.messages.any { it.id==m.id } }
        val repeated=restored.fingerprint in state.importedArchives
        dialog(if(repeated)"Backup already imported" else "Restore backup?").setOnCancelListener { restored.close() }.setMessage("${restored.state.messages.size} messages, ${restored.state.chats.size} private chats. $duplicate messages already exist and will be skipped. An empty workspace also restores preferences, approvals, and model weights.")
            .setNegativeButton("Cancel") { _,_->restored.close() }.setPositiveButton("Restore · skip duplicates") { _,_->work("Backup restored",onFailure={ restored.close() }) {
                try {
                    BackupArchive.attachments(restored.state).forEach { attachment ->
                        if(!vault.hasMedia(attachment))restored.media.getValue(attachment.file).inputStream().use { vault.importMedia(it,attachment.name,attachment.mime,attachment.duration,attachment.file) }
                    }
                    engine!!.state=BackupArchive.merge(engine!!.state,restored); configure()
                } finally { restored.close() }
            } }.showTracked()
    }
    private fun newChat() = chatNameSheet(null)
    private fun chatNameSheet(chat: Chat?) {
        val sheet=BottomSheetDialog(this)
        val content=column().apply { tag="chat-name-form"; setPadding(dp(20),dp(12),dp(20),dp(24)) }
        content.addView(View(this).apply { background=rounded(colors.divider,2) },LinearLayout.LayoutParams(dp(32),dp(4)).apply { gravity=Gravity.CENTER_HORIZONTAL; bottomMargin=dp(12) })
        content.addView(row().apply {
            addView(label(if(chat==null)"New private chat" else "Rename chat",22,ink,true),LinearLayout.LayoutParams(0,-2,1f))
            addView(icon("close","Close chat form") { sheet.dismiss() })
        })
        content.addView(label("Messages in this chat are not classified.",14,muted).apply { setPadding(0,dp(4),0,dp(20)) })
        val field=com.google.android.material.textfield.TextInputLayout(this).apply {
            hint="Chat name"; boxBackgroundMode=com.google.android.material.textfield.TextInputLayout.BOX_BACKGROUND_OUTLINE
            setBoxCornerRadii(dp(12).toFloat(),dp(12).toFloat(),dp(12).toFloat(),dp(12).toFloat())
            boxStrokeColor=green; boxBackgroundColor=colors.surface; isErrorEnabled=true
        }
        val name=com.google.android.material.textfield.TextInputEditText(field.context).apply {
            setSingleLine(); setTextColor(ink); textSize=16f; setText(chat?.title.orEmpty()); setSelectAllOnFocus(chat!=null)
            filters=arrayOf(android.text.InputFilter.LengthFilter(60))
            inputType=android.text.InputType.TYPE_CLASS_TEXT or android.text.InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
            imeOptions=android.view.inputmethod.EditorInfo.IME_ACTION_DONE
            contentDescription="Chat name"
        }
        field.addView(name); content.addView(field,LinearLayout.LayoutParams(-1,-2))
        val buttons=row().apply { gravity=Gravity.END; setPadding(0,dp(12),0,0) }
        buttons.addView(action("Cancel",{ sheet.dismiss() },false))
        val submit=action(if(chat==null)"Create chat" else "Save",{
            val title=name.text?.toString().orEmpty().trim()
            if(title.isBlank())field.error="Enter a chat name"
            else if(shown?.chats?.any { it.id!=chat?.id && it.title.equals(title,true) }==true)field.error="That chat name is already used"
            else {
                val changed=chat?.copy(title=title) ?: Chat(UUID.randomUUID().toString(),title)
                work("",onSuccess={ sheet.dismiss(); if(chat==null)navigate("chat:${changed.id}") }) {
                    if(chat==null)engine!!.addChat(changed)
                    else engine!!.state=engine!!.state.copy(chats=engine!!.state.chats.map { if(it.id==chat.id)changed else it })
                }
            }
        }).apply { isEnabled=name.text?.isNotBlank()==true; alpha=if(isEnabled)1f else .45f }
        buttons.addView(submit,LinearLayout.LayoutParams(-2,-2).apply { leftMargin=dp(8) }); content.addView(buttons)
        name.addTextChangedListener(watcher { field.error=null; submit.isEnabled=it.isNotBlank(); submit.alpha=if(submit.isEnabled)1f else .45f })
        name.setOnEditorActionListener { _,action,_ -> if(action==android.view.inputmethod.EditorInfo.IME_ACTION_DONE) { submit.performClick(); true } else false }
        sheet.setContentView(content); trackDialog(sheet); sheet.show()
        if(::vault.isInitialized && vault.lockedStorage)sheet.window?.addFlags(WindowManager.LayoutParams.FLAG_SECURE)
        sheet.window?.setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE)
        name.requestFocus(); name.post { (getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager).showSoftInput(name,InputMethodManager.SHOW_IMPLICIT) }
    }
    private fun privateChatMenu(chat: Chat) {
        menu(chat.title,listOf("Rename","Make a category","Change icon","Add to home screen","Forward chat","Delete chat messages")) { choice -> when(choice) {
            0->chatNameSheet(chat)
            1->{ val description=EditText(this).apply { hint="What belongs in this category?" }; dialog("Make ${chat.title} a category?").setMessage("Existing messages become confirmed examples. Use a unique category name.").setView(description)
                .setNegativeButton("Cancel",null).setPositiveButton("Add to categories") { _,_->work("Category created",onSuccess={ navigate(chat.id) }) { engine!!.promoteChat(chat.id,description.text.toString().trim()); configure(); maybeTrain() } }.showTracked() }
            2->categoryIconMenu(Category(chat.id,chat.title,"",icon=chat.icon,photo=chat.photo))
            3->pinShortcut("chat:${chat.id}",chat.title,chat.icon,chat.photo)
            4->forwardMessages(shown!!.messages.filter { it.visible() && it.chatId==chat.id })
            5->dialog("Delete this chat?").setMessage("Its messages go to Recently deleted. The chat remains available for recovered messages.").setNegativeButton("Cancel",null).setPositiveButton("Delete messages") { _,_->confirmDelete(shown!!.messages.filter { it.chatId==chat.id && it.visible() }.map { it.id }.toSet()) }.showTracked()
        } }
    }
    private fun pinShortcut(id: String,title: String,icon: String?,photo: Attachment?) {
        if(!ChatShortcuts.pin(this,id,title,CategoryIcons.resource(id,icon),photo?.let { imageCache.get(it.file) }))tell("This launcher does not support pinned shortcuts")
    }
    private fun forwardMessages(rows: List<Message>) {
        if(rows.isEmpty())return
        val state=shown ?: return
        val targets=listOf("all" to "Inbox")+state.categories.map { it.id to it.title }+state.chats.map { "chat:${it.id}" to it.title }
        menu("Forward ${rows.size} messages",targets.map { it.second }+"WhatsApp") { index ->
            if(index==targets.size)forwardWhatsApp(rows)
            else {
                var copiedId: String?=null
                clearSelection()
                work("",onSuccess={ navigate(targets[index].first); pendingJump=copiedId; highlightJump=true; render() }) {
                    val copies=engine!!.forward(rows.map { it.id }.toSet(),targets[index].first)
                    copiedId=copies.firstOrNull()?.id; configure(); maybeTrain()
                }
            }
        }
    }
    private fun forwardWhatsApp(rows: List<Message>) {
        selected.clear()
        work("",false) {
            val folder=File(cacheDir,"shared").apply { mkdirs() }
            val text=rows.joinToString("\n\n") { it.text.ifBlank { it.attachment?.name.orEmpty() } }
            val files=mutableListOf<File>()
            if(rows.size>20 || text.length>12000)File(folder,"chat-${UUID.randomUUID()}.txt").apply { writeText(ChatExport.render("Chat",rows,engine!!.state.categories,false)); files.add(this) }
            else rows.mapNotNull { it.attachment }.distinctBy { it.file }.forEach { attachment ->
                val file=File(folder,"${attachment.file}-${attachment.name.replace(Regex("[^A-Za-z0-9._-]"),"_")}")
                vault.readMedia(attachment).use { input->file.outputStream().use { input.copyTo(it) } }; files.add(file)
            }
            runOnUiThread { if(resumed) {
                val uris=ArrayList(files.map { FileProvider.getUriForFile(this,"$packageName.files",it) })
                val intent=Intent(if(uris.size>1)Intent.ACTION_SEND_MULTIPLE else Intent.ACTION_SEND).apply {
                    type=if(files.isEmpty())"text/plain" else if(files.size==1 && files[0].extension=="txt")"text/plain" else "*/*"
                    if(text.length<=12000)putExtra(Intent.EXTRA_TEXT,text)
                    if(uris.size==1)putExtra(Intent.EXTRA_STREAM,uris[0]) else if(uris.isNotEmpty())putParcelableArrayListExtra(Intent.EXTRA_STREAM,uris)
                    if(uris.isNotEmpty()) { clipData=ClipData.newRawUri("Forward",uris[0]).apply { uris.drop(1).forEach { addItem(ClipData.Item(it)) } }; addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION) }
                }
                try { startActivity(intent.setPackage("com.whatsapp")) } catch(_:android.content.ActivityNotFoundException) {
                    try { startActivity(intent.setPackage("com.whatsapp.w4b")) } catch(_:android.content.ActivityNotFoundException) { startActivity(Intent.createChooser(intent.setPackage(null),"Forward messages")) }
                }
            } }
        }
    }
    private fun widgetVisibility(enabled: Boolean, then: (() -> Unit)? = null) {
        val save={ work("",onSuccess=then) { engine!!.state=engine!!.state.copy(preferences=engine!!.state.preferences.copy(widgetWhenLocked=enabled)) } }
        if(enabled)dialog("Show tasks on the home screen?").setMessage("The widget will show task text even when Note Chat is locked. Tapping a task opens the app to unlock and update it.")
            .setNegativeButton("Cancel",null).setPositiveButton("Show tasks") { _,_->save() }.showTracked()
        else save()
    }
    private fun pinMessage(message: Message) {
        menu("Pin message for",listOf("24 hours","7 days","30 days","Forever")) { index ->
            val until=if(index==3)Long.MAX_VALUE else System.currentTimeMillis()+listOf(1L,7L,30L)[index]*ARCHIVE_DELAY_MS
            work("") { engine!!.pin(message.id,until) }
        }
    }
    private fun pinnedBar(current: List<Message>) {
        val pins=current.filter { (it.pinnedUntil ?: 0)>System.currentTimeMillis() }
        if(pins.isEmpty())return
        pinCursor=pinCursor.mod(pins.size)
        val pin=pins[pinCursor]
        val bar=row().apply { setPadding(dp(8),0,dp(8),0); setBackgroundColor(colors.inset); minimumHeight=dp(48) }
        bar.addView(icon("pin","Next pinned message") { pinCursor=(pinCursor+1).mod(pins.size); render() })
        bar.addView(label(pin.text.ifBlank { pin.attachment?.name.orEmpty() },14,ink).apply {
            maxLines=1; ellipsize=android.text.TextUtils.TruncateAt.END
            setOnClickListener { jumpTo(pin.id,true) }
        },LinearLayout.LayoutParams(0,-2,1f))
        bar.addView(label("${pinCursor+1}/${pins.size}",12,muted).apply { setPadding(dp(8),0,dp(8),0); setOnClickListener { showPins() } })
        bar.contentDescription="Pinned message. Tap to jump."
        if(selected.isNotEmpty())bar.visibility=View.INVISIBLE
        bar.setOnClickListener { jumpTo(pin.id,true) }
        root.addView(bar)
    }
    private fun showPins() {
        val pins=messages(shown ?: return,false).filter { (it.pinnedUntil ?: 0)>System.currentTimeMillis() }
        if(pins.isEmpty()) { toast("No pinned messages in this chat"); return }
        menu("Pinned messages",pins.map { it.text.ifBlank { it.attachment?.name.orEmpty() }.take(120) }) { index -> jumpTo(pins[index].id,true) }
    }
    private fun categoryIconMenu(category: Category) {
        menu("${category.title} icon",listOf("Material icon","Choose photo","Reset icon")) { choice -> when(choice) {
            0->{ val options=CategoryIcons.choices.keys.toList(); menu("Material icon",options.map { it.replaceFirstChar(Char::uppercase) }) { index -> work("") { engine!!.state=engine!!.state.copy(categories=engine!!.state.categories.map { if(it.id==category.id)it.copy(icon=options[index],photo=null) else it },chats=engine!!.state.chats.map { if(it.id==category.id)it.copy(icon=options[index],photo=null) else it }) } } }
            1->{ pendingCategoryPhoto=category.id; startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply { type="image/*"; addCategory(Intent.CATEGORY_OPENABLE) },14) }
            2->work("") { engine!!.state=engine!!.state.copy(categories=engine!!.state.categories.map { if(it.id==category.id)it.copy(icon=null,photo=null) else it },chats=engine!!.state.chats.map { if(it.id==category.id)it.copy(icon=null,photo=null) else it }) }
        } }
    }
    private fun categoryAvatar(id: String, state: ChatState): View {
        val category=state.categories.firstOrNull { it.id==id }
        val chat=state.chats.firstOrNull { "chat:${it.id}"==id }
        val background=GradientDrawable().apply { shape=GradientDrawable.OVAL; setColor(if(id=="all")green else Color.rgb(96,151,130)) }
        val avatar=FrameLayout(this).apply { this.background=background; clipToOutline=true; contentDescription="${category?.title ?: id} chat icon" }
        val image=ImageView(this).apply { scaleType=ImageView.ScaleType.CENTER_CROP }
        val photo=category?.photo ?: chat?.photo
        if(photo!=null) { image.tag="category:${category?.id ?: chat!!.id}"; loadImage(Message(id="category:${category?.id ?: chat!!.id}",attachment=photo),image); avatar.addView(image,FrameLayout.LayoutParams(-1,-1)) }
        else { image.setImageResource(CategoryIcons.resource(id,category?.icon ?: chat?.icon)); image.setColorFilter(Color.WHITE); avatar.addView(image,FrameLayout.LayoutParams(dp(28),dp(28),Gravity.CENTER)) }
        return avatar
    }
    private fun work(message:String="Saving…", refresh:Boolean=true, onFailure:(()->Unit)?=null, onSuccess:(()->Unit)?=null, action:()->Unit) {
        if(busy) { toast("Please wait for the current action to finish"); return }
        if(shown==null) return
        if(recorder!=null) { toast("Send or cancel the recording first"); return }
        val generation=sessionGeneration
        draft=editor?.text?.toString() ?: draft
        val storedDraft=draft
        val storedReply=reply
        rememberDraft()
        val ui=sessionState(shown!!)
        busy=true
        val progress=ProgressBar(this,null,android.R.attr.progressBarStyleHorizontal).apply { isIndeterminate=true }
        root.addView(progress,0,LinearLayout.LayoutParams(-1,dp(3)))
        worker.execute {
            val before=engine?.state
            try {
                val current=engine ?: error("Unlock the app first")
                action()
                if(before!=null)current.state=Reminders.infer(before,current.state)
                current.state=current.state.copy(draft=storedDraft,replyDraft=storedReply,editDraftId=ui.editDraftId,preEditDraft=ui.preEditDraft,drafts=ui.drafts,locations=ui.locations,guide=ui.guide,homeTab=ui.homeTab)
                vault.save(current.state)
                markdownDirty=true
                configure()
                publishWidget(current.state)
                TodoWidget.refresh(this)
                Reminders.sync(this,current.state)
                val state=current.state
                runOnUiThread { root.removeView(progress); busy=false; if (!isFinishing && generation==sessionGeneration && (!vault.lockedStorage || resumed)) { shown=projectPending(state); onSuccess?.invoke(); if(refresh) render(); scheduleIdleSave(); if(message!="Saving…" && message.isNotBlank()) toast(message) } }
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
        pendingText?.let { text->navigate("all"); draft=text.take(20000); pendingText=null; render() }
        pendingUri?.let { uri -> pendingUri=null; val mime=contentResolver.getType(uri).orEmpty(); val name=contentResolver.query(uri,arrayOf(OpenableColumns.DISPLAY_NAME),null,null,null)?.use { if(it.moveToFirst())it.getString(0) else "" }.orEmpty(); if(isZip(uri))importZip(uri) else if(mime.startsWith("text/") || name.endsWith(".md",true) || name.endsWith(".txt",true))importText(uri) else importUri(uri) }
        pendingTextUri?.let { uri->pendingTextUri=null; if(isZip(uri))importZip(uri) else importText(uri) }
        pendingFolder?.let { uri->pendingFolder=null; connectFolder(uri) }
    }
    private fun isNotebook(screen: String)=screen in setOf("all","starred","unsorted","todos") || shown?.categories?.any { it.id==screen }==true || screen.startsWith("chat:")
    private fun rememberDraft() {
        if(isNotebook(renderedRoute.ifBlank { route }))drafts[renderedRoute.ifBlank { route }]=ComposerDraft(editor?.text?.toString() ?: draft,reply,editTarget,editPreviousDraft,editor?.selectionEnd ?: -1)
    }
    private fun restoreDraft(screen: String) {
        val saved=drafts[screen] ?: ComposerDraft()
        pendingCaret=saved.cursor.takeIf { it>=0 }; draft=saved.text; reply=saved.replyTo; editTarget=saved.editId?.takeIf { id->shown?.messages?.any { it.id==id && it.visible() }==true }; editPreviousDraft=saved.previous
    }
    private fun sessionState(state: ChatState): ChatState = state.copy(navigation=positions[route] ?: Navigation(route),draft=draft,replyDraft=reply,editDraftId=editTarget,preEditDraft=editPreviousDraft,drafts=drafts.toMap(),locations=positions.toMap(),guide=guideProgress,homeTab=homeTab)
    private fun saveSession() {
        rememberDraft()
        val snapshot=sessionState(shown ?: return)
        worker.execute { engine?.let { current ->
            current.state=current.state.copy(navigation=snapshot.navigation,draft=snapshot.draft,replyDraft=snapshot.replyDraft,editDraftId=snapshot.editDraftId,preEditDraft=snapshot.preEditDraft,drafts=snapshot.drafts,locations=snapshot.locations,guide=snapshot.guide,homeTab=snapshot.homeTab)
            runCatching { persistState(current.state) }
        } }
    }
    private fun navigate(destination:String) {
        if(recorder!=null) { toast("Send or cancel the recording first"); return }
        if(destination!=route) {
            rememberDraft()
            cancelSplit(false)
            stopPlayback()
            restoreDraft(destination)
        }
        selected.clear(); searchMode=false; query=""; pendingJump=null
        route=destination
        if(destination in setOf("home","chats"))homeTab=destination
        render()
        saveSession()
    }
    private fun render() {
        val state=shown ?: return
        val keepFocus=editor?.hasFocus()==true
        val cursor=editor?.selectionEnd ?: 0
        applyAppearance()
        renderedArchives=archiveIds(state); renderedPins=pinIds(state)
        if(renderedRoute.isNotBlank() && (list!=null || root.childCount>0))positions[renderedRoute]=capturePosition()
        val changed=route!=renderedRoute
        if(!changed && editor!=null && pendingCaret==null)pendingCaret=cursor
        val backwards=route in setOf("home","settings")
        root.removeAllViews(); editor=null; list=null; viewport=null; overflowAnchor=null; headerView=null
        splitLines=null; composerContent=null; recordButton=null; recordingClock=null; recordingWave=null; modelStatus=null; voiceViews.clear()
        when(route) {
            "home", "chats" -> home(state)
            "settings" -> settings(state)
            "mods" -> modsScreen(state)
            "guide" -> guideScreen()
            "quiz" -> quizScreen(state)
            "settings:display","settings:data","settings:privacy","settings:archive" -> settingsDetails(state)
            "model" -> modelScreen(state)
            "categories" -> categories(state)
            "deleted" -> deleted(state)
            "completed", "archived-stars" -> archived(state)
            else -> conversation(state)
        }
        guidePanel()
        restorePosition(state)
        if(!changed && keepFocus && selected.isEmpty() && !searchMode)editor?.let { it.requestFocus(); it.setSelection(cursor.coerceIn(0,it.length())) }
        if(changed && renderedRoute.isNotEmpty() && android.animation.ValueAnimator.areAnimatorsEnabled())for(i in 0 until root.childCount)root.getChildAt(i).apply {
            alpha=0f; translationX=dp(if(backwards)-20 else 28).toFloat()
            animate().alpha(1f).translationX(0f).setDuration(190).setInterpolator(DecelerateInterpolator()).start()
        }
        renderedRoute=route
    }
    private fun home(state:ChatState) {
        val privateTab=route=="chats"
        homeTab=route
        val bar=row().apply { setPadding(dp(16),dp(8),dp(8),dp(8)) }
        bar.addView(label(if(privateTab)"Chats" else if(state.preferences.frontend=="notes")"All notes" else "Note Chat",25,if(state.preferences.frontend=="notes")ink else green,true),LinearLayout.LayoutParams(0,dp(48),1f))
        bar.addView(icon("add",if(privateTab)"New private chat" else "Add category") { if(privateTab)newChat() else editCategory(null) }.apply { setTag(R.id.guide_target,if(privateTab)"new-chat" else "add-category") })
        bar.addView(icon("settings","Settings") { navigate("settings") })
        bar.addView(icon("more","More options") { homeMenu() }.also { overflowAnchor=it })
        root.addView(bar)
        searchBox()
        if(state.preferences.frontend=="notes") {
            val boards=NoteBoard(this,state,colors,markdown,privateTab,query,{ target -> if(target=="new-chat")newChat() else navigate(target) },{ target ->
                when {
                    target.startsWith("chat:")->state.chats.firstOrNull { "chat:${it.id}"==target }?.let(::privateChatMenu)
                    state.categories.any { it.id==target }->categoryMenu(state.categories.first { it.id==target })
                    else->navigate(target)
                }
            })
            list=boards
            root.addView(HomeSwipe(this) { direction->switchHomeTab(direction) }.apply { addView(boards) },LinearLayout.LayoutParams(-1,0,1f))
            homeTabs(privateTab)
            return
        }
        val active=state.messages.filter { it.visible() }
        val inbox=active.filter { it.inInbox(state.preferences) }
        val rows=if(privateTab)state.chats.map { c->Triple("chat:${c.id}",c.title,active.filter { it.chatId==c.id }) }.toMutableList()
            else mutableListOf(Triple("all","Inbox",inbox),Triple("starred","Starred",active.filter { it.starred && !it.starArchived() })).apply {
                addAll(state.categories.sortedByDescending { it.pinned }.map { c->Triple(c.id,c.title,active.filter { it.chatId==null && it.category==c.id }) })
                add(Triple("unsorted","Unsorted",active.filter { it.chatId==null && !it.approved }))
            }
        fun filtered()=if(query.isBlank())rows else rows.filter { it.second.contains(query,true) }+active.filter { m->(if(privateTab)m.chatId!=null else m.inInbox(state.preferences)) && (m.text.contains(query,true) || m.attachment?.name?.contains(query,true)==true) }.map { m->Triple("message:${m.id}",state.chats.firstOrNull { it.id==m.chatId }?.title ?: state.categories.firstOrNull { it.id==m.category }?.title ?: "Inbox",listOf(m)) }
        val recycler=recycler()
        recycler.adapter=object:RecyclerView.Adapter<RowHolder>() {
            override fun getItemCount()=filtered().size
            override fun onCreateViewHolder(parent:ViewGroup,type:Int)=RowHolder(column())
            override fun onBindViewHolder(holder:RowHolder,position:Int) {
                val (id,title,messages)=filtered()[position]
                holder.box.removeAllViews()
                holder.box.tag=id
                val item=row().apply { setPadding(dp(16),dp(10),dp(14),dp(10)); minimumHeight=dp(82) }
                item.addView(categoryAvatar(id,state),LinearLayout.LayoutParams(dp(54),dp(54)).apply { rightMargin=dp(14) })
                val middle=column()
                middle.addView(label(title,17,ink,true))
                val last=messages.lastOrNull()
                middle.addView(label(last?.text?.ifBlank { last.attachment?.name.orEmpty() } ?: if(id=="all")"Send yourself a message" else "No messages yet",14,muted).apply { maxLines=1; ellipsize=android.text.TextUtils.TruncateAt.END; setPadding(0,dp(6),0,0) })
                item.addView(middle,LinearLayout.LayoutParams(0,-2,1f))
                val right=column().apply { gravity=Gravity.END }
                right.addView(label(last?.let { time(it.created) } ?: "",11,muted))
                if(!privateTab && state.categories.any { it.id==id } && messages.isNotEmpty())right.addView(label(messages.size.toString(),11,muted).apply { setTypeface(null,Typeface.ITALIC); setPadding(0,dp(6),0,0) })
                item.addView(right)
                item.setOnClickListener {
                    if(id.startsWith("message:"))jumpTo(id.removePrefix("message:"),true)
                    else navigate(id)
                }
                item.setOnLongClickListener { if(id.startsWith("chat:"))privateChatMenu(state.chats.first { "chat:${it.id}"==id }) else state.categories.firstOrNull { it.id==id }?.let(::categoryMenu); true }
                holder.box.addView(item)
                holder.box.addView(View(this@MainActivity).apply { setBackgroundColor(colors.divider) },LinearLayout.LayoutParams(-1,dp(1)).apply { leftMargin=dp(84) })
            }
        }
        list=recycler
        root.addView(HomeSwipe(this) { direction->switchHomeTab(direction) }.apply { addView(recycler) },LinearLayout.LayoutParams(-1,0,1f))
        if(privateTab && rows.isEmpty())root.addView(action("Create a private chat",::newChat,false).apply { setTag(R.id.guide_target,"new-chat") })
        homeTabs(privateTab)
    }
    private fun switchHomeTab(direction: Int) {
        if(route=="home" && direction>0)navigate("chats") else if(route=="chats" && direction<0)navigate("home")
    }
    private fun homeTabs(privateTab: Boolean) {
        val tabs=row().apply { setPadding(dp(12),dp(4),dp(12),dp(4)); setBackgroundColor(colors.surface) }
        listOf("home" to "Notes","chats" to "Chats").forEach { (target,name) ->
            val active=if(target=="chats")privateTab else !privateTab
            val tab=column().apply { gravity=Gravity.CENTER; setPadding(0,dp(5),0,dp(5)); contentDescription="$name tab"; setOnClickListener { navigate(target) }; setTag(R.id.guide_target,if(target=="chats")"chats-tab" else "notes-tab") }
            tab.addView(ImageView(this).apply {
                setImageDrawable(ChatIcon(if(target=="chats")"chat" else "notes",if(active)green else muted))
                setPadding(dp(14),dp(4),dp(14),dp(4))
                if(active)background=rounded(colors.inset,16)
            },LinearLayout.LayoutParams(dp(52),dp(32)))
            tab.addView(label(name,12,if(active)ink else muted,active).apply { gravity=Gravity.CENTER; setPadding(0,dp(4),0,0) })
            tabs.addView(tab,LinearLayout.LayoutParams(0,dp(66),1f))
        }
        root.addView(tabs)
    }
    private fun homeMenu() { overflow(listOf("New private chat","Import TXT / Markdown / ZIP","Reclassify all","Starred notes","Manage categories","Settings")) { when(it) { 0->newChat(); 1->pickText(); 2->reclassifyAll(); 3->navigate("starred"); 4->navigate("categories"); 5->navigate("settings") } } }
    private fun messages(state:ChatState,includeSearch:Boolean=true)=state.messages.filter { it.visible() && when(route) {
        "all" -> it.inInbox(state.preferences); "starred" -> it.starred && !it.starArchived(); "unsorted" -> it.chatId==null && !it.approved; else -> if(route.startsWith("chat:"))it.chatId==route.removePrefix("chat:") else it.chatId==null && it.category==route
    } }.filter { !includeSearch || query.isBlank() || it.text.contains(query,true) || it.attachment?.name?.contains(query,true)==true }
    private fun title(state:ChatState)=when(route) { "all"->"Inbox"; "starred"->"Starred"; "unsorted"->"Unsorted"; else->state.chats.firstOrNull { "chat:${it.id}"==route }?.title ?: state.categories.firstOrNull { it.id==route }?.title ?: "Inbox" }
    private fun conversation(state:ChatState) {
        if(route=="todos" && !searchMode) { todoList(state); return }
        val current=messages(state,false)
        selected.retainAll(current.map { it.id }.toSet())
        if(selected.isNotEmpty()) selectionToolbar(state)
        else if(searchMode) searchToolbar(current)
        else if(state.preferences.frontend=="notes" && editTarget!=null)toolbar(title(state),"Editing note",{ cancelEdit() },listOf("check" to { saveEdit() }))
        else toolbar(title(state),"",{ navigate(homeTab) },listOf("search" to { beginSearch() },"more" to { chatMenu() }))
        if(!searchMode)pinnedBar(current)
        val recycler=recycler().apply { setBackgroundColor(if(state.preferences.frontend=="notes")colors.surface else paper) }
        val manager=recycler.layoutManager as LinearLayoutManager
        manager.stackFromEnd=false
        recycler.adapter=MessageRows(this,current) { box,message,position ->
            val live=shown ?: state
            val records=messages(live,false)
            box.setBackgroundColor(if(message.id in selected)colors.selected else Color.TRANSPARENT)
            if(live.preferences.frontend!="notes" && (position==0 || day(message.created)!=day(records.getOrNull(position-1)?.created ?: message.created))) {
                val date=row().apply { gravity=Gravity.CENTER; setPadding(0,dp(8),0,dp(8)) }
                date.addView(pill(day(message.created),muted,colors.inset)); box.addView(date)
            }
            box.addView(messageBubble(live,message))
            if(live.preferences.frontend=="notes")box.addView(View(this).apply { setBackgroundColor(colors.divider) },LinearLayout.LayoutParams(-1,dp(1)).apply { leftMargin=dp(24); rightMargin=dp(24) })
            viewport?.bind(box,message.id)
        }
        list=recycler; viewport=ChatViewport(this,recycler,current.map { it.id },colors)
        root.addView(viewport,LinearLayout.LayoutParams(-1,0,1f))
        if(current.isNotEmpty() && state.preferences.frontend!="notes")recycler.scrollToPosition(current.lastIndex)
        if(current.isEmpty())root.addView(label(if(route=="unsorted")"No messages awaiting approval" else "Send yourself a message",13,muted).apply { gravity=Gravity.CENTER; setPadding(dp(16),dp(8),dp(16),dp(8)) })
        if(searchMode)searchNavigation(current)
        if(state.preferences.frontend!="notes" || editTarget==null)composer(state)
        if(selected.isNotEmpty())composerRoot?.visibility=View.INVISIBLE
    }
    private fun messageBubble(state:ChatState,message:Message,recovering:Boolean=false):View {
        val notes=state.preferences.frontend=="notes" && !recovering
        val width=(resources.displayMetrics.widthPixels-dp(if(notes)48 else if(recovering)60 else 64)).coerceAtLeast(dp(180))
        val wrapper: FrameLayout=if(notes)FrameLayout(this) else ReplySwipe(this) { if(!recovering)replyTo(message) }
        wrapper.apply {
            setPadding(dp(if(notes)20 else 8),dp(if(notes)8 else 3),dp(if(notes)20 else 8),dp(if(notes)8 else 3))
            if(this is ReplySwipe)enabledForReply=!recovering && selected.isEmpty()
            layoutParams=LinearLayout.LayoutParams(-1,-2)
            if(!recovering) { setOnClickListener { if(selected.isNotEmpty())toggleSelection(message.id) }; setOnLongClickListener { toggleSelection(message.id); true } }
        }
        val body=BubbleLayout(this@MainActivity,width).apply { orientation=LinearLayout.VERTICAL; background=if(notes)null else rounded(bubble,10); setPadding(dp(if(notes)4 else 9),dp(6),dp(if(notes)4 else 9),dp(4)); minimumWidth=dp(150) }
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
        if(notes && editTarget==message.id)body.addView(EditText(this).apply {
            textSize=18f; setTextColor(ink); background=null; setText(draft); pendingCaret?.let { caret->setSelection(caret.coerceIn(0,length())); pendingCaret=null }; minLines=3
            inputType=android.text.InputType.TYPE_CLASS_TEXT or android.text.InputType.TYPE_TEXT_FLAG_MULTI_LINE or android.text.InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
            setPadding(0,dp(8),0,dp(8)); setTag(R.id.guide_target,"composer")
            addTextChangedListener(watcher { draft=it }); editor=this
        },LinearLayout.LayoutParams(-1,-2))
        else if(message.text.isNotBlank())body.addView(MessageText(this@MainActivity) { if(!recovering)tagging(message) }.apply {
            textSize=if(notes)18f else 16f; setTextColor(ink); immediateTap=notes || selected.isNotEmpty(); setPadding(0,dp(3),0,dp(5)); maxWidth=width-dp(18)
            interceptTap={ if(selected.isNotEmpty()) { toggleSelection(message.id); true } else false }
            linkMenu=::linkActions
            markdown.render(this,message.text,if(searchMode && !recovering)query else "",::openLink)
            setOnClickListener { if(selected.isNotEmpty())toggleSelection(message.id) else if(notes)editMessage(message) }
            if(!recovering)setOnLongClickListener { toggleSelection(message.id); true }
        })
        message.translation?.takeIf { it!=message.text }?.let { translated -> body.addView(label("Translation\n$translated",14,muted).apply { setPadding(0,dp(6),0,dp(6)) }) }
        val footer=row().apply { gravity=Gravity.CENTER_VERTICAL }
        val category=state.chats.firstOrNull { it.id==message.chatId }?.title ?: state.categories.firstOrNull { it.id==message.category }?.title ?: "Tag"
        footer.addView(label(category,if(notes)12 else 11,green).apply { setTag(R.id.guide_target,"tag"); setPadding(0,dp(4),dp(10),dp(4)); if(!recovering)setOnClickListener { if(selected.isNotEmpty())toggleSelection(message.id) else tagging(message) } },LinearLayout.LayoutParams(0,-2,1f))
        footer.addView(label("${if(message.starred)"★ " else ""}${if(message.edited)"edited · " else ""}${if(notes)day(message.created)+" · "+time(message.created) else time(message.created)}${if(message.approved)" ✓✓" else " ✓"}",10,muted))
        body.addView(footer)
        if(recovering) {
            val days=((RETENTION_MS-(System.currentTimeMillis()-message.deleted!!))/(24*60*60*1000)).coerceAtLeast(0)
            body.addView(label("Deleted ${day(message.deleted)} · $days days left",11,muted).apply { setPadding(0,dp(4),0,0) })
            body.addView(row().apply { gravity=Gravity.END; addView(action("Restore",{ restoreMessage(message) },false)) })
        }
        val feedback=row().apply { gravity=Gravity.END; setTag(R.id.guide_target,"feedback") }
        if(!recovering && message.category!=null && message.chatId==null && state.preferences.classification) {
            feedback.addView(action(if(message.approved)"👍 ✓" else "👍",{ approveMessages(setOf(message.id)) },false).apply { contentDescription=if(message.trainable())"Approve category and teach the classifier" else "Approve category" })
            if(message.trainable())feedback.addView(action("👎",{ disapprove(message) },false).apply { contentDescription="Try another category without learning" })
        }
        if(feedback.childCount>0)body.addView(feedback.apply { visibility=if(selected.isEmpty())View.VISIBLE else View.INVISIBLE })
        body.setOnClickListener { if(selected.isNotEmpty())toggleSelection(message.id) else if(state.preferences.frontend=="notes")editMessage(message) }
        if(!recovering)body.setOnLongClickListener { toggleSelection(message.id); true }
        if(!recovering && !notes) {
            val content=row().apply { gravity=Gravity.CENTER_VERTICAL }
            content.addView(forwardButton(message),LinearLayout.LayoutParams(dp(40),dp(40)).apply { rightMargin=dp(6) })
            content.addView(body,LinearLayout.LayoutParams(-2,-2))
            wrapper.addView(content,FrameLayout.LayoutParams(-2,-2,Gravity.END))
        } else {
            if(notes)footer.addView(forwardButton(message),0,LinearLayout.LayoutParams(dp(40),dp(40)).apply { rightMargin=dp(6) })
            wrapper.addView(body,FrameLayout.LayoutParams(if(notes)-1 else -2,-2,if(notes)Gravity.START else Gravity.END))
        }
        return wrapper
    }
    private fun forwardButton(message: Message)=ImageButton(this).apply {
        contentDescription="Forward message"
        setImageDrawable(ChatIcon("forward",muted))
        background=rounded(colors.inset,20)
        setPadding(dp(10),dp(10),dp(10),dp(10))
        setTag(R.id.guide_target,"forward")
        setOnClickListener { forwardAction(if(selected.isNotEmpty())shown!!.messages.filter { it.id in selected } else listOf(message)) }
        setOnLongClickListener { toggleSelection(message.id); true }
    }
    private fun composer(state:ChatState) {
        val box=column().apply { setBackgroundColor(paper); setPadding(dp(7),dp(5),dp(7),dp(7)) }
        composerRoot=box
        if(editTarget!=null)box.addView(row().apply { addView(label("Edit message",14,green),LinearLayout.LayoutParams(0,-2,1f)); addView(icon("close","Cancel edit") { cancelEdit() }) })
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
            setOnClickListener { if(recorder!=null)stopRecording(true) else if(draft.isNotBlank()) { if(editTarget!=null)saveEdit() else if(splitTarget!=null)sendSplitPart() else sendText(draft) } else if(recordVoice()) { recordingLocked=true; updateRecordingComposer() } }
            gesturesEnabled=draft.isBlank() && recorder==null && splitTarget==null
            startHold={ recordVoice() }; releaseHold={ stopRecording(true) }; cancelHold={ stopRecording(false) }; lockHold={ recordingLocked=true; updateRecordingComposer() }
        }
        recordButton=button
        line.addView(button,LinearLayout.LayoutParams(dp(48),dp(48)).apply { leftMargin=dp(6) }); box.addView(line); root.addView(box)
        if(recorder!=null)updateRecordingComposer() else {
            val inputRow=row().apply { background=rounded(colors.surface,if(state.preferences.frontend=="notes")10 else 26); setPadding(dp(4),0,dp(3),0) }
            inputRow.addView(icon("attach","Attach or import") { attachMenu() }.apply { isEnabled=splitTarget==null; alpha=if(isEnabled)1f else .35f })
            val input=EditText(this).apply { hint=if(state.preferences.frontend=="notes")"Write a note" else "Message"; textSize=16f; setTextColor(ink); setHintTextColor(muted); background=null; setText(draft); maxLines=5; setPadding(dp(4),dp(9),dp(4),dp(9)) }
            input.setTag(R.id.guide_target,"composer"); pendingCaret?.let { caret->input.setSelection(caret.coerceIn(0,input.length())); pendingCaret=null }; editor=input; inputRow.addView(input,LinearLayout.LayoutParams(0,-2,1f)); inputRow.addView(icon("camera","Choose image") { pickImage() }.apply { isEnabled=splitTarget==null; alpha=if(isEnabled)1f else .35f }); content.addView(inputRow)
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
        if(text.isBlank() || text.length>20000) { tell("Messages must contain text and fit within 20,000 characters"); return }
        val category=shown?.categories?.firstOrNull { it.id==route }?.id
        val chat=route.removePrefix("chat:").takeIf { route.startsWith("chat:") }
        val message=Message(text=text.trim(),replyTo=reply,category=category,chatId=chat,manual=category!=null,approved=category!=null)
        followLatest=list?.canScrollVertically(1)!=true
        draft=""; reply=null; editor?.setText("")
        val add: (ChatEngine)->Unit={ current -> current.state=current.state.copy(messages=current.state.messages+message) }
        fast(train=category!=null,success={ if(shown?.preferences?.autoTranslate==true)translateMessage(message,false) },preview=add,commit={ current ->
            val suggestion=if(category!=null)Suggestion(category,"Confirmed by sending to this category") else predict(message)
            val scheduled=if(current.state.preferences.reminders && suggestion.category=="todos")ReminderTime.detect(message.text) else null
            current.state=current.state.copy(messages=current.state.messages+message.copy(category=suggestion.category,reason=suggestion.reason,remindAt=scheduled))
        })
    }
    private fun clearSelection() { val changed=selected.toSet(); selected.clear(); refreshSelection(changed) }
    private fun toggleSelection(id:String) {
        val changed=selected.toSet()+id
        if(id in selected)selected.remove(id) else selected.add(id)
        draft=editor?.text?.toString() ?: draft
        if(searchMode) { searchMode=false; query=""; render() } else refreshSelection(changed)
    }
    private fun refreshSelection(changed: Set<String>) {
        val state=shown ?: return
        if(list?.adapter !is MessageRows || headerView==null) { render(); return }
        val index=root.indexOfChild(headerView)
        root.removeView(headerView)
        if(selected.isNotEmpty())selectionToolbar(state) else toolbar(title(state),"",{ navigate(homeTab) },listOf("search" to { beginSearch() },"more" to { chatMenu() }))
        val next=headerView ?: return; root.removeView(next); root.addView(next,index.coerceAtLeast(0))
        composerRoot?.visibility=if(selected.isEmpty())View.VISIBLE else View.INVISIBLE
        (list?.adapter as? MessageRows)?.selectionChanged(changed)
        (0 until root.childCount).map { root.getChildAt(it) }.firstOrNull { it.contentDescription=="Pinned message. Tap to jump." }?.visibility=if(selected.isEmpty())View.VISIBLE else View.INVISIBLE
    }
    private fun selectionToolbar(state:ChatState) {
        val bar=row().apply { setPadding(dp(4),dp(4),dp(4),dp(4)); setBackgroundColor(colors.surface) }
        bar.addView(icon("back","Clear selection") { clearSelection() })
        bar.addView(label(selected.size.toString(),20,ink,true),LinearLayout.LayoutParams(0,-2,1f))
        if(selected.size==1)bar.addView(icon("reply","Reply") { state.messages.firstOrNull { it.id in selected }?.let { replyTo(it) } })
        bar.addView(icon("star","Star selected messages") { val ids=selected.toSet(); selected.clear(); work("") { ids.forEach { id->engine!!.update(id) { it.copy(starred=true,starArchiveAt=null) } } } })
        bar.addView(icon("copy","Copy selected messages") { copyText(state.messages.filter { it.id in selected }.joinToString("\n\n") { it.text.ifBlank { it.attachment?.name.orEmpty() } }); clearSelection() })
        bar.addView(icon("delete","Delete selected messages") { confirmDelete(selected.toSet()) })
        bar.addView(icon("more","Selected message options") { selectionMenu(state) }.also { overflowAnchor=it })
        root.addView(bar); headerView=bar
    }
    private fun selectionMenu(state:ChatState) {
        val rows=state.messages.filter { it.id in selected }
        val options=linkedMapOf<String,()->Unit>()
        options["Tag messages"]={ tagSelection() }
        if(rows.any { it.category!=null && it.chatId==null })options["Approve selected"]={ approveMessages(rows.filter { it.category!=null && it.chatId==null }.map { it.id }.toSet()) }
        options["Forward"]={ forwardMessages(rows) }
        options["Share"]={ if(rows.size==1 && rows[0].attachment!=null)shareAttachment(rows[0]) else if(rows.any { it.attachment!=null })shareSelection(rows) else shareText(rows.joinToString("\n\n") { it.text }) }
        options["Pinned messages"]={ clearSelection(); showPins() }
        options["Select all"]={ val ids=messages(state,false).map { it.id }.toSet(); val changed=selected.toSet()+ids; selected.addAll(ids); refreshSelection(changed) }
        options["Unstar"]={ val ids=selected.toSet(); clearSelection(); work("") { ids.forEach { id->engine!!.update(id) { it.copy(starred=false,starArchiveAt=null) } } } }
        if(rows.any { it.starred })options["Archive starred in 24 hours"]={ val ids=rows.filter { it.starred }.map { it.id }.toSet(); clearSelection(); work("") { engine!!.archiveStars(ids) } }
        if(rows.size==1) {
            val message=rows.first()
            options[if((message.pinnedUntil ?: 0)>System.currentTimeMillis())"Unpin message" else "Pin message"]={ clearSelection(); if((message.pinnedUntil ?: 0)>System.currentTimeMillis())work("") { engine!!.pin(message.id,null) } else pinMessage(message) }
            options["Edit"]={ clearSelection(); editMessage(message) }
            if(message.attachment==null)options["Split message"]={ startSplit(message) }
            if(message.attachment?.mime?.startsWith("audio/")==true)options["Transcribe voice message"]={ clearSelection(); transcribe(message) }
            if(message.attachment?.mime?.startsWith("image/")==true)options["Extract text (OCR)"]={ clearSelection(); extractText(message) }
            if(message.text.isNotBlank())options["Translate"]={ clearSelection(); translateMessage(message) }
            options["Set reminder"]={ clearSelection(); setReminder(message) }
            if(message.remindAt!=null)options["Remove reminder"]={ clearSelection(); fast(preview={ it.update(message.id) { record->record.copy(remindAt=null) } }) }
            options["Why this tag?"]={ tell(message.reason.ifBlank { "Manually tagged attachment." }) }
        } else if(rows.size>1 && rows.all { it.attachment==null })options["Merge messages"]={
            dialog("Merge ${rows.size} messages?").setMessage("Keep the first message's position and time. Replies will point to the merged message. The combined text gets a new category suggestion.")
                .setNegativeButton("Cancel",null).setPositiveButton("Merge") { _,_-> val ids=rows.map { it.id }.toSet(); clearSelection(); work("") { val merged=engine!!.merge(ids); configure(); suggest(merged); pendingJump=merged.id } }.showTracked()
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
    private fun jumpTo(id:String,highlight:Boolean=false) {
        val state=shown ?: return
        val message=state.messages.firstOrNull { it.id==id && it.visible() } ?: return
        if(messages(state,false).none { it.id==id })navigate(message.chatId?.let { "chat:$it" } ?: "all")
        pendingJump=id; highlightJump=highlight; render()
    }
    private fun tagging(m:Message) { chooseTags(setOf(m.id)) }
    private fun tagSelection() { chooseTags(selected.toSet()) }
    private fun chooseTags(ids:Set<String>) {
        val categories=shown?.categories ?: return
        menu("Tag ${if(ids.size==1)"message" else "messages"}",categories.map { it.title }+"Unsorted") { i-> val changed=selected.toSet(); selected.clear(); refreshSelection(changed); fast(train=true,preview={ current -> ids.forEach { current.tag(it,categories.getOrNull(i)?.id) } }) }
    }
    private fun editMessage(m:Message) {
        selected.clear(); editPreviousDraft=editor?.text?.toString() ?: draft; editTarget=m.id; draft=m.text
        if(messages(shown!!,false).none { it.id==m.id })route=m.chatId?.let { "chat:$it" } ?: "all"
        pendingCaret=m.text.length
        pendingJump=m.id
        render()
        root.post { editor?.let { field -> field.requestFocus(); (getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager).showSoftInput(field,InputMethodManager.SHOW_IMPLICIT) } }
    }
    private fun cancelEdit() { editTarget=null; draft=editPreviousDraft; editPreviousDraft=""; render() }
    private fun saveEdit() {
        val id=editTarget ?: return; val text=editor?.text?.toString() ?: draft
        if(text.isBlank())return
        editTarget=null; draft=editPreviousDraft; editPreviousDraft=""; editor=null
        fast(train=true,preview={ it.edit(id,text) },commit={ current -> current.edit(id,text); configure(); val m=current.state.messages.first { it.id==id }; if(m.trainable())suggest(m) })
        render()
    }
    private fun chatMenu() {
        val options=linkedMapOf<String,()->Unit>()
        options["Search"]={ beginSearch() }; options["Import TXT / Markdown / ZIP"]={ pickText() }
        options["Select messages"]={ messages(shown!!,false).lastOrNull()?.let { toggleSelection(it.id) } }
        options["Export chat"]={ exportMenu() }; options["Forward chat"]={ forwardMessages(messages(shown!!,false)) }
        options["Go to first message"]={ messages(shown!!,false).firstOrNull()?.let { jumpTo(it.id) } }
        options["Pinned messages"]={ showPins() }
        options["Add to home screen"]={ val state=shown!!; pinShortcut(route,title(state),state.categories.firstOrNull { it.id==route }?.icon,state.categories.firstOrNull { it.id==route }?.photo) }
        if(route.startsWith("chat:"))options["Chat settings"]={ shown!!.chats.firstOrNull { "chat:${it.id}"==route }?.let(::privateChatMenu) }
        else options["Manage categories"]={ navigate("categories") }
        options["Delete chat messages"]={ confirmDelete(messages(shown!!,false).map { it.id }.toSet()) }
        overflow(options.keys.toList()) { options.values.elementAt(it)() }
    }
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
        if(value.startsWith("notechat-md:")) { openMarkdownLink(Uri.decode(value.removePrefix("notechat-md:"))); return }
        if(!value.contains(":") && value.substringBefore('#').endsWith(".md",true)) { openMarkdownLink(Uri.decode(value)); return }
        val link=if(value.contains(":") && !value.startsWith("www."))value else "https://$value"
        val uri=Uri.parse(link)
        if(uri.scheme !in listOf("https","http","mailto","tel")) { tell("This link type cannot be opened"); return }
        try { startActivity(Intent(Intent.ACTION_VIEW,uri)) } catch(_:Exception) { tell("No app can open this link") }
    }
    private fun linkActions(link:String) { menu(link,listOf("Open link","Copy link","Share link")) { when(it) { 0->openLink(link); 1->copyText(link); 2->shareText(link) } } }
    private fun copyText(text:String) { (getSystemService(CLIPBOARD_SERVICE) as ClipboardManager).setPrimaryClip(ClipData.newPlainText("Message",text)) }
    private fun shareText(text:String) { startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).apply { type="text/plain"; putExtra(Intent.EXTRA_TEXT,text) },"Share")) }
    private fun watcher(changed:(String)->Unit)=object:TextWatcher { override fun beforeTextChanged(s:CharSequence?,start:Int,count:Int,after:Int)=Unit; override fun onTextChanged(s:CharSequence?,start:Int,before:Int,count:Int) { lastInteraction=android.os.SystemClock.uptimeMillis(); if(markdownDirty)scheduleIdleSave(); changed(s.toString()) }; override fun afterTextChanged(s:Editable?)=Unit }
    private fun settings(state:ChatState) {
        val content=settingsBody("Settings")
        setting(content,"Display",if(state.preferences.frontend=="notes")"Notes · Inbox and two-column boards" else "Chat layout") { navigate("settings:display") }
        setting(content,"Storage & backups","Permanent Markdown folder, ZIP backups, Drive, and imports") { navigate("settings:data") }
        setting(content,"Classification & learning",if(state.preferences.classification)"On · personal learner and model tests" else "Off · manual organization") { navigate("model") }
        setting(content,"MODs","Optional transcription, translation, OCR, and reminders") { navigate("mods") }
        setting(content,"Privacy & security","App lock and widget visibility") { navigate("settings:privacy") }
        setting(content,"Archives & deletion","Completed tasks, archived stars, and Recently deleted") { navigate("settings:archive") }
        setting(content,"User guide","Walk through capture, corrections, private chats, backups, and MODs") { navigate("guide") }
    }
    private fun settingsBody(title:String): LinearLayout {
        toolbar(title,"",{ navigate(if(route=="settings")"home" else "settings") })
        val scroll=ScrollView(this); val content=column().apply { setPadding(dp(16),dp(12),dp(16),dp(24)) }
        scroll.addView(content); root.addView(scroll,LinearLayout.LayoutParams(-1,0,1f)); return content
    }
    private fun settingsDetails(state:ChatState) {
        val content=settingsBody(when(route) { "settings:display"->"Display"; "settings:data"->"Storage & backups"; "settings:privacy"->"Privacy & security"; else->"Archives & deletion" })
        when(route) {
            "settings:display"->{
                setting(content,"Layout",if(state.preferences.frontend=="notes")"Notes · Inbox and two-column boards" else "Chat") { menu("Layout",listOf("Chat","Notes")) { choice->work("") { engine!!.state=engine!!.state.copy(preferences=engine!!.state.preferences.copy(frontend=if(choice==0)"chat" else "notes")) } } }
                setting(content,"Appearance",state.preferences.appearance.replaceFirstChar { it.uppercase() }) { menu("Appearance",listOf("System","Light","Dark")) { index->work("") { engine!!.state=engine!!.state.copy(preferences=engine!!.state.preferences.copy(appearance=listOf("system","light","dark")[index])) } } }
                setting(content,"Include private chats in Inbox",if(state.preferences.includePrivateChatsInInbox)"On · notes and private messages together" else "Off · private messages stay in Chats") { work("") { engine!!.state=engine!!.state.copy(preferences=engine!!.state.preferences.copy(includePrivateChatsInInbox=!state.preferences.includePrivateChatsInInbox)) } }
                setting(content,"To-do widget","Add a checklist to your Android home screen") { requestTodoWidget() }
            }
            "settings:data"->{
                setting(content,"Permanent Markdown folder",markdownError ?: if(state.markdown.tree==null)"Choose a folder. Files survive uninstall until you delete the folder." else "Connected · saves after 30 seconds idle or when you leave. Files survive uninstall.") { markdownSettings() }
                content.addView(label("Markdown saves run in the background after 30 seconds without activity and when you leave the app. Sync now is available in the folder menu. Folder copies are readable. Category contents follow edits; app deletion still uses 90-day recovery. The saved workspace and attachments remain outside the app until you remove them.",13,muted).apply { setPadding(dp(4),dp(8),dp(4),dp(12)) })
                setting(content,"Create encrypted backup","ZIP · messages, media, private chats, approvals, and personal head. Save to device or Drive.") { createBackup() }
                setting(content,"Import / restore ZIP","Open a backup from device or Drive, or a WhatsApp export. Duplicate message IDs are detected.") { pickText() }
                setting(content,"Import TXT / Markdown","Preview messages before importing") { pickText() }
                setting(content,"Keep imported dates",if(state.preferences.preserveImportDates)"On · transcript dates" else "Off · use import time") { work("") { engine!!.state=engine!!.state.copy(preferences=engine!!.state.preferences.copy(preserveImportDates=!state.preferences.preserveImportDates)) } }
                setting(content,"Categories","Add, rename, change icons, or remove categories") { navigate("categories") }
            }
            "settings:privacy"->{
                setting(content,"App lock",if(vault.lockedStorage)"Fingerprint or screen lock · locks when you leave" else "Off · private storage remains encrypted") {
                    if(vault.lockedStorage)work("App lock disabled") { vault.disableLock(); runOnUiThread { privacy() } }
                    else try { authenticate(vault.enableLockCipher()) { cipher->work("App lock enabled") { vault.finishWrapping(cipher,true); runOnUiThread { privacy() } } } } catch(_:Exception) { tell("Set up a fingerprint or device screen lock first") }
                }
                setting(content,"Show widget tasks while locked",if(state.preferences.widgetWhenLocked)"On · task text is visible on the home screen" else "Off · widget asks to unlock") { widgetVisibility(!state.preferences.widgetWhenLocked) }
                content.addView(label("Notes are not sent to a classification server. Drive backups and WhatsApp forwarding use only destinations you choose. Optional Google ML Kit MODs process content on-device and send SDK performance/usage metrics to Google.",13,muted))
            }
            else->{
                setting(content,"Completed to-dos","${state.messages.count { it.deleted==null && it.completedArchived() }} archived · completed tasks move here after one day") { navigate("completed") }
                setting(content,"Archived starred messages","${state.messages.count { it.deleted==null && it.starArchived() }} messages") { navigate("archived-stars") }
                setting(content,"Recently deleted","${state.messages.count { it.deleted!=null }} messages · recover for 90 days") { navigate("deleted") }
                setting(content,"Delete by date range","Choose the start and end date/time") { deleteDateRange() }
            }
        }
    }
    private fun guideScreen() {
        val content=settingsBody("User guide")
        content.addView(label("Learn in the app",24,ink,true))
        content.addView(label("The walkthrough keeps its instructions on the real screens and highlights each control. Your drafts, editing state, and scroll positions stay with their notebooks. It uses your notes and changes things only when you choose an action.",16,muted).apply { setPadding(0,dp(16),0,dp(24)) })
        content.addView(action(if(guideProgress.step>0 || guideProgress.active)"Resume walkthrough" else "Start walkthrough",{ startGuide() }))
        if(guideProgress.step>0)content.addView(action("Start from the beginning",{ startGuide(true) },false))
        GuidePages.steps.forEachIndexed { index,step -> setting(content,"${index+1}. ${step.title}",step.text) { startGuide(step=index) } }
    }
    private fun startGuide(restart: Boolean=false, step: Int?=null) {
        val returning=if(guideProgress.active || guideProgress.step>0)guideProgress.returnTo else positions["settings"] ?: Navigation("settings")
        guideProgress=GuideProgress(true,step ?: if(restart)0 else guideProgress.step,returning,if(guideProgress.active || guideProgress.step>0)guideProgress.returnTab else homeTab)
        showGuideStep()
    }
    private fun showGuideStep() {
        guideReveal=true
        navigate(GuidePages.steps[guideProgress.step].route)
    }
    private fun moveGuide(delta: Int) {
        if(recorder!=null) { toast("Send or cancel the recording before the next step"); return }
        if(delta>0 && guideProgress.step==GuidePages.steps.lastIndex) { pauseGuide(true); return }
        guideProgress=guideProgress.copy(step=(guideProgress.step+delta).coerceIn(0,GuidePages.steps.lastIndex))
        showGuideStep()
    }
    private fun pauseGuide(finished: Boolean=false) {
        val returning=guideProgress.returnTo
        homeTab=guideProgress.returnTab
        guideProgress=guideProgress.copy(active=false,step=if(finished)0 else guideProgress.step)
        positions[returning.route]=returning
        navigate(returning.route)
    }
    private fun descendants(view: View): Sequence<View> = sequence {
        yield(view)
        if(view is ViewGroup)for(index in 0 until view.childCount)yieldAll(descendants(view.getChildAt(index)))
    }
    private fun guidePanel() {
        guideAnchor?.foreground=guideForeground
        guideAnchor=null; guideForeground=null
        if(!guideProgress.active || route=="guide")return
        val coach=GuideCoach(this,guideProgress,colors,{ moveGuide(-1) },::showGuideStep,{ moveGuide(1) },{ pauseGuide() })
        val params=LinearLayout.LayoutParams(-1,-2).apply { leftMargin=dp(8); rightMargin=dp(8); topMargin=dp(4); bottomMargin=dp(4) }
        val composerIndex=composerRoot?.let { root.indexOfChild(it) } ?: -1
        if(composerIndex>=0)root.addView(coach,composerIndex,params) else root.addView(coach,params)
        val step=guideProgress.step
        val screen=route
        root.doOnLayout {
            if(!guideProgress.active || guideProgress.step!=step || route!=screen)return@doOnLayout
            val target=GuidePages.steps[step].target
            val anchor=descendants(root).firstOrNull { it.getTag(R.id.guide_target)==target }
                ?: if(target in setOf("feedback","tag","task"))descendants(root).firstOrNull { it.getTag(R.id.guide_target)=="composer" } else null
            if(anchor!=null) {
                guideAnchor=anchor; guideForeground=anchor.foreground
                anchor.foreground=GradientDrawable().apply { setColor(Color.TRANSPARENT); setStroke(dp(2),green); cornerRadius=dp(10).toFloat() }
                if(guideReveal) { guideReveal=false; anchor.requestRectangleOnScreen(android.graphics.Rect(0,0,anchor.width,anchor.height),false) }
            }
        }
    }
    private fun refreshBoards() {
        val state=shown ?: return
        val old=list as? NoteBoard ?: return
        val container=old.parent as? ViewGroup ?: return
        val replacement=NoteBoard(this,state,colors,markdown,route=="chats",query,{ target->if(target=="new-chat")newChat() else navigate(target) },{ target->state.chats.firstOrNull { "chat:${it.id}"==target }?.let(::privateChatMenu) ?: state.categories.firstOrNull { it.id==target }?.let(::categoryMenu) })
        container.removeView(old); container.addView(replacement); list=replacement
    }
    private fun runMod(title: String, queued: Boolean=false, action: ()->Unit) {
        if(modBusy && !queued) { toast("A MOD is already working"); return }
        queuedMods++; modBusy=true; if(!queued)toast(title)
        modWorker.execute { try { action() } catch(error:Exception) { runOnUiThread { if(resumed)tell(error.cause?.message ?: error.message ?: "MOD failed") } }
            finally { runOnUiThread { queuedMods--; modBusy=queuedMods>0 } } }
    }
    private fun modConsent(title: String, then: ()->Unit) {
        dialog(title).setMessage("Google ML Kit processes text/images on this phone. It downloads models and sends SDK usage/performance metrics to Google. Translation language downloads use Wi-Fi.")
            .setNegativeButton("Cancel",null).setPositiveButton("Enable") { _,_->then() }.showTracked()
    }
    private fun modsScreen(state: ChatState) {
        val content=settingsBody("MODs")
        setting(content,"Voice transcription",if(VoiceTranscriber.installed(this))"English model installed · transcribe from message options" else "Optional Vosk download · English · 40 MB") { speechSettings() }
        setting(content,"OCR · image text",if(state.preferences.ocr)"Enabled · Latin script · extract from image options" else "Off · optional Google Play services model download") {
            if(state.preferences.ocr)work("") { engine!!.state=engine!!.state.copy(preferences=engine!!.state.preferences.copy(ocr=false)) }
            else modConsent("Enable OCR?") { runMod("Downloading OCR model…") { Mods.installOcr(this); runOnUiThread { if(resumed)work("OCR enabled") { engine!!.state=engine!!.state.copy(preferences=engine!!.state.preferences.copy(ocr=true)) } } } }
        }
        setting(content,"Automatic translation",if(state.preferences.autoTranslate)"On · original text is kept · target ${Locale.forLanguageTag(state.preferences.targetLanguage).displayLanguage}" else "Off · downloads language models as needed over Wi-Fi") {
            val enable={ work("") { engine!!.state=engine!!.state.copy(preferences=engine!!.state.preferences.copy(autoTranslate=!state.preferences.autoTranslate)) } }
            if(state.preferences.autoTranslate)enable() else modConsent("Enable automatic translation?") { enable() }
        }
        setting(content,"Translation language",Locale.forLanguageTag(state.preferences.targetLanguage).displayLanguage) { val languages=Mods.languages(); menu("Translate to",languages.map { Locale.forLanguageTag(it).displayLanguage }) { index->work("") { engine!!.state=engine!!.state.copy(preferences=engine!!.state.preferences.copy(targetLanguage=languages[index])) } } }
        setting(content,"Translation downloads","Download or remove language models · roughly 30 MB per language") { translationDownloads() }
        setting(content,"Reminders",if(state.preferences.reminders)"On · manual reminders and detected to-do times" else "Off · enable notifications for reminders") {
            if(!state.preferences.reminders && android.os.Build.VERSION.SDK_INT>=33 && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)!=PackageManager.PERMISSION_GRANTED)requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS),21)
            else work("") { engine!!.state=engine!!.state.copy(preferences=engine!!.state.preferences.copy(reminders=!state.preferences.reminders)) }
        }
        setting(content,"Reminder notifications","Android notification permissions and alert settings") { startActivity(Intent(android.provider.Settings.ACTION_APP_NOTIFICATION_SETTINGS).putExtra(android.provider.Settings.EXTRA_APP_PACKAGE,packageName)) }
        setting(content,"MiniLM sentence similarity",if(MiniLm.modelFile(this).exists())"Downloaded · optional classifier encoder" else "Optional 23 MB download · configure under Model & learning") { navigate("model") }
        content.addView(label("OCR and translation require Google ML Kit; OCR and language identification require Google Play services. Transcription uses Vosk. The base classifier works without these MODs. Automatic time detection supports English dates/times; battery saving can delay reminders.",13,muted).apply { setPadding(0,dp(16),0,0) })
    }
    private fun translationDownloads() {
        modConsent("Manage translation models?") { runMod("Reading language models…") {
            val downloaded=Mods.downloaded()
            runOnUiThread { if(resumed)menu("Translation models",listOf("Download a language")+downloaded.map { "Remove ${Locale.forLanguageTag(it).displayLanguage}" }) { choice ->
                if(choice==0) { val languages=Mods.languages().filter { it!="en" }; menu("Download language · Wi-Fi",languages.map { Locale.forLanguageTag(it).displayLanguage }) { index->runMod("Downloading language…") { Mods.download(languages[index]); runOnUiThread { toast("Language model downloaded") } } } }
                else runMod("Removing language…") { Mods.remove(downloaded.elementAt(choice-1)); runOnUiThread { toast("Language model removed") } }
            } }
        } }
    }
    private fun translateMessage(message: Message, ask: Boolean=true) {
        val state=shown ?: return
        val run={ runMod("Translating on this phone…",queued=!ask) {
            val translated=Mods.translate(this,message.text,state.preferences.targetLanguage)
            runOnUiThread { if(resumed && shown?.messages?.any { it.id==message.id && it.text==message.text }==true)fast(preview={ current -> current.update(message.id) { it.copy(translation=translated) } }) }
        } }
        if(ask && !state.preferences.autoTranslate)modConsent("Translate this message?") { run() } else run()
    }
    private fun extractText(message: Message) {
        if(shown?.preferences?.ocr!=true) { navigate("mods"); toast("Enable OCR before extracting image text"); return }
        val attachment=message.attachment ?: return
        val file=File(cacheDir,"ocr-${UUID.randomUUID()}")
        worker.execute { try {
            vault.readMedia(attachment).use { input->file.outputStream().use { input.copyTo(it) } }
            runOnUiThread { if(resumed && !modBusy)runMod("Extracting image text…") {
                val text=try { Mods.ocr(file) } finally { file.delete() }
                runOnUiThread { if(resumed) {
                    if(text.isBlank())tell("No readable text found") else {
                        val input=EditText(this).apply { setText(text.take(20000)); maxLines=8 }
                        dialog("Review extracted text").setView(input).setNegativeButton("Cancel",null).setPositiveButton("Save text") { _,_->
                            val reviewed=input.text.toString()
                            fast(train=true,preview={ current->current.update(message.id) { it.copy(text=reviewed,extracted=true,approved=false,edited=true) } },commit={ current ->
                                current.update(message.id) { it.copy(text=reviewed,extracted=true,approved=false,edited=true,category=null,rejected=emptySet()) }; configure(); suggest(current.state.messages.first { it.id==message.id })
                            })
                        }.showTracked()
                    }
                } }
            } else file.delete() }
        } catch(error:Exception) { file.delete(); runOnUiThread { tell(error.message ?: "Cannot read image") } } }
    }
    private fun setReminder(message: Message) {
        if(shown?.preferences?.reminders!=true) { navigate("mods"); toast("Enable Reminders first"); return }
        val selected=java.util.Calendar.getInstance().apply { timeInMillis=message.remindAt ?: System.currentTimeMillis()+60*60*1000 }
        val date=android.app.DatePickerDialog(this,{ _,year,month,day ->
            selected.set(year,month,day)
            val time=android.app.TimePickerDialog(this,{ _,hour,minute ->
                selected.set(java.util.Calendar.HOUR_OF_DAY,hour); selected.set(java.util.Calendar.MINUTE,minute); selected.set(java.util.Calendar.SECOND,0); selected.set(java.util.Calendar.MILLISECOND,0)
                if(selected.timeInMillis<=System.currentTimeMillis())tell("Choose a future time") else fast(preview={ current->current.update(message.id) { it.copy(remindAt=selected.timeInMillis) } })
            },selected.get(java.util.Calendar.HOUR_OF_DAY),selected.get(java.util.Calendar.MINUTE),true)
            trackDialog(time); time.show()
        },selected.get(java.util.Calendar.YEAR),selected.get(java.util.Calendar.MONTH),selected.get(java.util.Calendar.DAY_OF_MONTH))
        trackDialog(date); date.show()
    }
    private fun startQuiz() {
        if(shown?.preferences?.classification!=true) { tell("Turn classification on to test the model"); return }
        val input=EditText(this).apply { hint="10 fresh notes, one per line"; minLines=6; maxLines=10 }
        dialog("Test with 10 notes").setMessage("Predictions are frozen before you choose labels. Use notes the model has not already learned.").setView(input)
            .setNegativeButton("Cancel",null).setPositiveButton("Predict") { _,_->
                val examples=input.text.toString().lines().map(String::trim).filter(String::isNotBlank)
                if(examples.size!=10 || examples.distinctBy(::normalized).size!=10 || examples.any { it.length>20000 }) { tell("Enter ten different notes, one per line"); return@setPositiveButton }
                work("",onSuccess={ navigate("quiz") }) {
                    configure()
                    val results=examples.map { text -> val start=System.nanoTime(); val category=predict(Message(text=text)).category; QuizItem(text,category,nanos=System.nanoTime()-start) }
                    engine!!.state=engine!!.state.copy(quiz=Quiz(results,android.os.Debug.getPss().coerceAtMost(Int.MAX_VALUE.toLong()).toInt()))
                }
            }.showTracked()
    }
    private fun quizScreen(state: ChatState) {
        val content=settingsBody("Test with 10 notes")
        val quiz=state.quiz ?: run { content.addView(action("Start a test",::startQuiz)); return }
        val answered=quiz.items.count { it.answer!=null }
        val score=Diagnostics.accuracy(quiz)
        content.addView(label("$answered/${quiz.items.size} labelled${score?.let { " · ${"%.0f".format(it*100)}% correct" } ?: ""}",22,ink,true))
        content.addView(label("Predictions were saved before your answers. This is a small personal check, not a general accuracy benchmark.",14,muted))
        quiz.items.forEachIndexed { index,item ->
            setting(content,"${index+1}. ${item.text}","Model: ${state.categories.firstOrNull { it.id==item.prediction }?.title ?: "Unsorted"} · Your label: ${state.categories.firstOrNull { it.id==item.answer }?.title ?: "Choose"}") {
                menu("Your category",state.categories.map { it.title }) { choice -> fast(preview={ current -> val active=current.state.quiz ?: return@fast; current.state=current.state.copy(quiz=active.copy(items=active.items.mapIndexed { i,row->if(i==index)row.copy(answer=state.categories[choice].id) else row })) }) }
            }
        }
        if(answered==quiz.items.size)content.addView(action("Save answers as approved notes",{
            work("Approved examples added") {
                val active=engine!!.state.quiz ?: return@work
                val incoming=active.items.map { Message(text=it.text,category=it.answer,approved=true,manual=true,reason="Confirmed in the model test") }
                engine!!.state=engine!!.state.copy(messages=engine!!.state.messages+incoming,quiz=null); configure(); maybeTrain()
            }
        }))
        content.addView(action("Start another test",::startQuiz,false))
        content.addView(label("Whole-app PSS when tested: ${quiz.memoryKb/1024} MB. Mean classification time: ${"%.2f".format(quiz.items.map { it.nanos/1_000_000.0 }.average())} ms.",13,muted))
    }
    private fun reclassifyAll() {
        if(shown?.preferences?.classification!=true) { toast("Classification is turned off"); return }
        val count=shown?.messages?.count { it.visible() && !it.approved && it.trainable() } ?: 0
        if(count==0) { toast("No unapproved text to reclassify"); return }
        dialog("Reclassify $count messages?").setMessage("Refresh unapproved text and reviewed voice transcripts, including manual tags. Approved messages and media without transcripts stay unchanged.")
            .setNegativeButton("Cancel",null).setPositiveButton("Reclassify") { _,_->
                work("Suggestions refreshed") { configure(); engine!!.reclassifyPending { predict(it) } }
            }.showTracked()
    }
    private fun modelScreen(state:ChatState) {
        toolbar("Model & learning","On this phone",{ navigate("settings") })
        val scroll=ScrollView(this); val content=column().apply { setPadding(dp(20),dp(12),dp(20),dp(24)) }; scroll.addView(content); root.addView(scroll,LinearLayout.LayoutParams(-1,0,1f))
        setting(content,"Automatic classification",if(state.preferences.classification)"On · only category / inbox messages are classified" else "Off · private chats and manual organization still work") {
            work("") {
                val enabled=!engine!!.state.preferences.classification
                engine!!.state=engine!!.state.copy(preferences=engine!!.state.preferences.copy(classification=enabled))
                encoder?.close(); encoder=null
                if(enabled && MiniLm.modelFile(this).exists())encoder=MiniLm(this)
                classifier=TextClassifier(encoder,Diagnostics::record); configure()
            }
        }
        content.addView(label(if(state.preferences.classification)"Local learner · installed" else "Local learner · paused",21,ink,true))
        content.addView(label("Learns from thumbs-up and manual category assignments with no download or account. Approved messages are remembered immediately. A small classifier head trains after enough examples.",15,muted).apply { setPadding(0,dp(8),0,dp(16)) })
        modelStatus=label("${if(busy)"Working…" else "Ready"} · ${ChatEngine(state).confirmed().size} distinct approved messages",14,green).also { content.addView(it) }
        content.addView(label("How learning works",16,ink,true).apply { setPadding(0,dp(22),0,dp(8)) })
        content.addView(ModelFlowView(this),LinearLayout.LayoutParams(-1,dp(112)))
        content.addView(label("Thumbs-up or a manual tag confirms your label. Thumbs-down only tries another tag. Reviewed voice transcripts and extracted image text can be classified.",14,muted))
        val counts=state.categories.map { c->c.title to (ChatEngine(state).confirmed().count { it.category==c.id }) }
        counts.forEach { (name,count)-> val line=row().apply { setPadding(0,dp(10),0,dp(10)) }; line.addView(label(name,15,ink),LinearLayout.LayoutParams(0,-2,1f)); line.addView(label("$count approved${if(count>=6)" ✓" else ""}",13,muted)); content.addView(line) }
        state.report?.let { report->content.addView(label("Latest evaluation: ${report.heldOut} held-out messages. Macro F1 ${"%.2f".format(report.candidate)}; starter ${"%.2f".format(report.baseline)}. ${if(report.adopted && state.preferences.classification && state.model?.let { TextClassifier.compatible(it,state.categories,encoder?.identity) }==true)"Personal head active." else if(report.adopted)"Personal head saved; enable classification and its matching encoder to use it." else "Starter retained."}",14,muted).apply { setPadding(0,dp(14),0,dp(14)) }) }
        setting(content,"Share model report","Inspect the learner without exporting messages, labels, examples, or weights") { exportModelReport() }
        content.addView(label("Train & compare uses approved notes, keeps held-out examples separate, and adopts a personal head only when its macro F1 improves. Test with 10 notes checks fresh examples against your own labels.",14,muted))
        setting(content,"Test with 10 notes","Enter fresh notes, then choose the right labels to measure accuracy") { startQuiz() }
        setting(content,"Export personal weights","Encrypted ZIP · excludes notes and correction memory. Treat trained weights as private.") { createBackup(true) }
        setting(content,"Import personal weights","Open a model ZIP with its password. Categories and feature encoder must match.") { pickText() }
        val diagnostics=Diagnostics.report(android.os.Debug.getPss().coerceAtMost(Int.MAX_VALUE.toLong()).toInt())
        content.addView(label(if(diagnostics.getInt("sampleCount")==0)"No classification timings yet this session. Send or reclassify a note to measure the classifier." else "${diagnostics.getInt("sampleCount")} session predictions · median ${"%.2f".format(diagnostics.getDouble("medianClassificationMs"))} ms · p95 ${"%.2f".format(diagnostics.getDouble("p95ClassificationMs"))} ms. Whole-app PSS: ${diagnostics.opt("appPssKb")} KB. Excludes UI and download time.",13,muted))
        content.addView(action("Train & compare",{ if(!state.preferences.classification) { tell("Turn classification on first"); return@action }; work("") { runOnUiThread { modelStatus?.text="Training and evaluating on this phone…" }; val result=classifier.train(); engine!!.state=engine!!.state.copy(report=result,model=classifier.learned(),splits=classifier.splits()) } }))
        setting(content,"MiniLM sentence encoder",if(MiniLm.modelFile(this).exists())"Installed · offline · INT8 · ${MiniLm.modelFile(this).length()/1024/1024} MB" else "Optional download · about 23 MB. Adds sentence similarity; the built-in learner already works.") {
            if(!MiniLm.modelFile(this).exists())dialog("Download MiniLM?").setMessage("Downloads public weights. Your notes stay on your phone.").setNegativeButton("Cancel",null).setPositiveButton("Download") { _,_->work("") {
                MiniLm.download(this) { percent->runOnUiThread { modelStatus?.text="Downloading MiniLM · $percent%" } }
                encoder=if(engine!!.state.preferences.classification)MiniLm(this) else null; modelFailure=null; classifier=TextClassifier(encoder,Diagnostics::record); configure()
            } }.showTracked() else menu("MiniLM",listOf("Remove downloaded encoder")) { work("") { encoder?.close(); encoder=null; MiniLm.modelFile(this).delete(); classifier=TextClassifier(null,Diagnostics::record); configure() } }
        }
        modelFailure?.let { content.addView(label("MiniLM could not start: $it. The built-in learner is active.",14,muted)) }
        content.addView(label("Encoder weights stay frozen. The personal head uses approved text; held-out labels stay outside training. Learning never runs on dislikes or predictions. No accuracy target is guaranteed; use fresh notes to measure progress.",13,muted))
    }
    private fun categories(state:ChatState) {
        toolbar("Categories","One chat per category",{ navigate("settings") },listOf("add" to { editCategory(null) }))
        (headerView as? ViewGroup)?.getChildAt((headerView as ViewGroup).childCount-1)?.setTag(R.id.guide_target,"add-category")
        val scroll=ScrollView(this); val content=column().apply { setPadding(dp(16),dp(10),dp(16),dp(20)) }; scroll.addView(content); root.addView(scroll,LinearLayout.LayoutParams(-1,0,1f))
        state.categories.forEach { c -> setting(content,c.title + if(c.pinned)" · pinned" else "",c.description) { categoryMenu(c) } }
        content.addView(action("Add category",{ editCategory(null) }))
    }
    private fun categoryMenu(c:Category) { menu(c.title,listOf("Edit category",if(c.pinned)"Unpin chat" else "Pin chat","Remove category","Export category","Change chat icon","Add to home screen")) { i -> when(i) {
        0->editCategory(c); 1->work { engine!!.state=engine!!.state.copy(categories=engine!!.state.categories.map { if(it.id==c.id)it.copy(pinned=!it.pinned) else it }) }
        2->{ val others=shown!!.categories.filterNot { it.id==c.id }; menu("Move its messages before removing",listOf("Unsorted")+others.map { it.title }) { target ->work("Category removed · messages kept") { engine!!.removeCategory(c.id,if(target==0)null else others[target-1].id); configure(); runOnUiThread { if(route==c.id)route="home" } } } }
        3->{ navigate(c.id); exportMenu() }; 4->categoryIconMenu(c); 5->pinShortcut(c.id,c.title,c.icon,c.photo)
    } } }
    private fun editCategory(c:Category?) {
        val box=column().apply { setPadding(dp(20),dp(4),dp(20),0) }
        val title=EditText(this).apply { hint="Name"; setText(c?.title.orEmpty()) }; val description=EditText(this).apply { hint="What belongs here?"; setText(c?.description.orEmpty()); maxLines=5 }
        val examples=EditText(this).apply { hint="Example messages (one per line)"; setText(c?.examples?.joinToString("\n").orEmpty()); maxLines=5 }
        box.addView(title); box.addView(description); box.addView(examples)
        dialog(if(c==null)"Add category" else "Edit category").setView(box).setNegativeButton("Cancel",null).setPositiveButton("Save") { _,_->
            val category=Category(c?.id ?: UUID.randomUUID().toString(),title.text.toString().trim(),description.text.toString().trim(),examples.text.toString().lines().map { it.trim() }.filter { it.isNotBlank() }.take(30),c?.pinned ?: false,c?.icon,c?.photo)
            work { if(c==null)engine!!.addCategory(category) else {
                require(category.title.isNotBlank() && category.title.length<=60 && category.description.isNotBlank() && category.description.length<=2000 && category.examples.all { it.length<=2000 })
                require(engine!!.state.categories.none { it.id!=category.id && it.title.equals(category.title,true) }) { "That category name is already used" }
                engine!!.state=engine!!.state.copy(categories=engine!!.state.categories.map { if(it.id==category.id)category else it },model=null,report=null,quiz=null)
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
    private fun pickText() { startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).apply { type="*/*"; putExtra(Intent.EXTRA_MIME_TYPES,arrayOf("text/plain","text/markdown","text/x-markdown","application/octet-stream","application/x-markdown","application/zip","application/x-zip-compressed")); addCategory(Intent.CATEGORY_OPENABLE) },12) }
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
        box.addView(label("${parsed.entries.size} messages will be added to Inbox${if(shown!!.preferences.classification)" with local suggestions" else " without classification"}. Imported labels are not approvals.",15,ink))
        box.addView(label(if(shown!!.preferences.preserveImportDates)"Transcript dates will be kept." else "Messages use the import date and time. Change this in Settings to keep transcript dates.",13,muted))
        val format=CheckBox(this).apply { text="Dates use day / month / year"; isChecked=dayFirst; visibility=if(parsed.transcript)View.VISIBLE else View.GONE }
        box.addView(format)
        dialog("Import ${if(parsed.transcript)"chat transcript" else "notes"}?").setView(box).setNegativeButton("Cancel",null).setPositiveButton("Import") { _,_->
            val entries=ChatImport.parse(source,format.isChecked).entries
            navigate("all")
            val importedAt=System.currentTimeMillis()
            work("") {
                val incoming=ChatImport.messages(entries,engine!!.state.preferences.preserveImportDates,importedAt).map { message ->
                    predict(message).let { message.copy(category=it.category,reason=it.reason) } }
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
        val photoCategory=restoredCategoryPhoto; restoredCategoryPhoto=null
        val category=shown?.categories?.firstOrNull { it.id==route }?.id
        val destinationChat=route.removePrefix("chat:").takeIf { route.startsWith("chat:") }
        if(photoCategory!=null) {
            work("Chat icon updated") {
                require(mime.startsWith("image/")) { "Choose an image" }
                val photo=contentResolver.openInputStream(uri)?.use { vault.importMedia(it,name,mime) } ?: error("Cannot read image")
                engine!!.state=engine!!.state.copy(categories=engine!!.state.categories.map { if(it.id==photoCategory)it.copy(photo=photo) else it },chats=engine!!.state.chats.map { if(it.id==photoCategory)it.copy(photo=photo) else it })
            }; return
        }
        val caption=editor?.text?.toString() ?: draft
        val shouldFollow=list?.canScrollVertically(1)!=true
        val replyId=reply; draft=""; editor?.setText(""); reply=null
        work(if(category==null)"Attachment saved · choose a category" else "",onFailure={ draft=caption; reply=replyId; render() }) {
            val mediaDuration=if(mime.startsWith("audio/"))try { android.media.MediaMetadataRetriever().let { reader->try { reader.setDataSource(this,uri); reader.extractMetadata(android.media.MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L } finally { reader.release() } } } catch(_:Exception) { 0L } else 0L
            val attachment=contentResolver.openInputStream(uri)?.use { vault.importMedia(it,name,mime,mediaDuration) } ?: error("Cannot read attachment")
            engine!!.state=engine!!.state.copy(messages=engine!!.state.messages+Message(text=caption,attachment=attachment,replyTo=replyId,category=category,manual=category!=null,approved=category!=null,chatId=destinationChat,reason="Manually tag this attachment."))
            followLatest=shouldFollow
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
        val destinationChat=route.removePrefix("chat:").takeIf { route.startsWith("chat:") }
            val replyId=reply; reply=null
            work("") { try { val attachment=file.inputStream().use { vault.importMedia(it,"Voice message.m4a","audio/mp4",elapsed) }.copy(waveform=wave); engine!!.state=engine!!.state.copy(messages=engine!!.state.messages+Message(attachment=attachment,replyTo=replyId,category=category,manual=category!=null,approved=category!=null,chatId=destinationChat,reason="Voice messages are manually tagged.")); followLatest=true } finally { file.delete() } }
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
            runOnUiThread { if(generation==sessionGeneration && resumed && view.tag==message.id && (shown?.messages?.any { it.id==message.id && (it.deleted==null || System.currentTimeMillis()-it.deleted<RETENTION_MS) }==true || shown?.categories?.any { "category:${it.id}"==message.id && it.photo?.file==a.file }==true || shown?.chats?.any { "category:${it.id}"==message.id && it.photo?.file==a.file }==true)){ imageCache.put(a.file,bitmap); view.setImageBitmap(bitmap) } else bitmap.recycle() }
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
        route=message.chatId?.let { "chat:$it" } ?: "all"; splitTarget=message.id; draft=message.text; reply=null; selected.clear()
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
        else { val done=shown!!.messages.first { it.id==message.id }.done; fast(preview={ it.complete(message.id,!done) }) }
    }
    private fun todoList(state: ChatState) {
        val current=messages(state,false)
        selected.retainAll(current.map { it.id }.toSet())
        if(selected.isNotEmpty())selectionToolbar(state) else toolbar("To-do","",{ navigate(homeTab) },listOf("search" to { beginSearch() },"more" to { chatMenu() }))
        pinnedBar(current)
        val recycler=recycler()
        recycler.adapter=MessageRows(this,current) { box,m,_ ->
            box.setBackgroundColor(if(m.id in selected)colors.selected else colors.surface)
            val line=TodoRow(this,m,colors,markdown,{ toggleTask(m) },{ toggleSelection(m.id) })
            line.text.linkMenu=::linkActions
            line.footer.addView(label(time(m.created),11,muted).apply { setOnClickListener { tagging(m) } },LinearLayout.LayoutParams(0,-2,1f))
            line.footer.addView(action("Tag",{ if(selected.isEmpty())tagging(m) else toggleSelection(m.id) },false))
            if(m.trainable()) {
                line.footer.addView(action(if(m.approved)"👍 ✓" else "👍",{ approveMessages(setOf(m.id)) },false).apply { visibility=if(selected.isEmpty())View.VISIBLE else View.INVISIBLE })
                line.footer.addView(action("👎",{ disapprove(m) },false).apply { visibility=if(selected.isEmpty())View.VISIBLE else View.INVISIBLE })
            }
            box.addView(line); box.tag=m.id
            box.addView(View(this).apply { setBackgroundColor(colors.divider) },LinearLayout.LayoutParams(-1,dp(1)))
        }
        list=recycler; viewport=ChatViewport(this,recycler,current.map { it.id },colors); root.addView(viewport,LinearLayout.LayoutParams(-1,0,1f))
        if(state.preferences.frontend!="notes" || editTarget==null)composer(state)
        if(selected.isNotEmpty())composerRoot?.visibility=View.INVISIBLE
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
        if(shown?.preferences?.widgetWhenLocked!=true) { widgetVisibility(true,::pinTodoWidget); return }
        pinTodoWidget()
    }
    private fun pinTodoWidget() {
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
        if(!current.state.preferences.classification)return
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
        if(resultCode!=RESULT_OK) { if(requestCode==11)pendingExport=null; if(requestCode==14)pendingCategoryPhoto=null; return }
        val uri=data?.data ?: return
        if(requestCode==13) {
            try {
                val access=Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION
                require((data.flags and access)==access) { "Choose a folder with read and write access" }
                contentResolver.takePersistableUriPermission(uri,Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION)
                pendingFolder=uri
                if(shown!=null)processPending()
            } catch(e:Exception) { tell(e.message ?: "Cannot keep folder access") }
            return
        }
        if(requestCode==14) { val id=pendingCategoryPhoto ?: return; pendingCategoryPhoto=null; restoredCategoryPhoto=id; pendingUri=uri; if(shown!=null)processPending(); return }
        if(requestCode==12) { pendingTextUri=uri; if(shown!=null)processPending() }
        if(requestCode==10) { pendingUri=uri; if(shown!=null)processPending() }
        if(requestCode==11) {
            val file=pendingExport ?: return; pendingExport=null
            worker.execute { try { contentResolver.openOutputStream(uri)?.use { output->file.inputStream().use { it.copyTo(output) } } ?: error("Cannot write chosen file"); runOnUiThread { toast("Chat exported") } } catch(e:Exception) { runOnUiThread { tell(e.message ?: "Export failed") } } }
        }
    }
    override fun onRequestPermissionsResult(requestCode:Int,permissions:Array<out String>,results:IntArray) {
        super.onRequestPermissionsResult(requestCode,permissions,results)
        if(requestCode==21) { if(results.firstOrNull()==PackageManager.PERMISSION_GRANTED && shown!=null)work("Reminders enabled") { engine!!.state=engine!!.state.copy(preferences=engine!!.state.preferences.copy(reminders=true)) } else tell("Reminder notifications are blocked. Allow them in Android settings to receive alerts.") }
        if(requestCode==20 && results.firstOrNull()==PackageManager.PERMISSION_GRANTED) { if(recordVoice()) { recordingLocked=true; updateRecordingComposer() } } else if(requestCode==20)tell("Microphone access is needed only to record a voice message.")
    }
    @Deprecated("Deprecated in Java") override fun onBackPressed() {
        if(recorder!=null) { stopRecording(false); return }
        if(editTarget!=null) { cancelEdit(); return }
        if(splitTarget!=null) { cancelSplit(); return }
        if(selected.isNotEmpty()) { clearSelection(); return }
        if(searchMode) { searchMode=false; query=""; render(); return }
        if(route in setOf("model","mods","guide","quiz") || route.startsWith("settings:")) { navigate("settings"); return }
        if(route !in setOf("home","chats")) { navigate(homeTab) } else if(route=="chats")navigate("home") else super.onBackPressed()
    }
    private fun searchBox() {
        val search=EditText(this).apply { hint="Search messages or categories"; textSize=15f; setSingleLine(); background=rounded(colors.inset,22); setText(query); setPadding(dp(18),dp(10),dp(18),dp(10)) }
        search.addTextChangedListener(object:TextWatcher { override fun beforeTextChanged(s:CharSequence?,start:Int,count:Int,after:Int)=Unit; override fun onTextChanged(s:CharSequence?,start:Int,before:Int,count:Int) { query=s.toString(); if(shown?.preferences?.frontend=="notes" && route in setOf("home","chats"))refreshBoards() else list?.adapter?.notifyDataSetChanged() }; override fun afterTextChanged(s:Editable?)=Unit })
        search.setOnEditorActionListener { _,_,_->(getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager).hideSoftInputFromWindow(search.windowToken,0); true }
        root.addView(search,LinearLayout.LayoutParams(-1,dp(46)).apply { leftMargin=dp(16); rightMargin=dp(16); bottomMargin=dp(8) })
    }
    private fun toolbar(title:String,subtitle:String,back:()->Unit,actions:List<Pair<String,()->Unit>> = emptyList()) {
        val bar=row().apply { setPadding(dp(4),dp(4),dp(4),dp(4)); setBackgroundColor(colors.surface) }
        bar.addView(icon("back","Back",back)); val titles=column(); titles.addView(label(title,19,ink,true)); if(subtitle.isNotBlank())titles.addView(label(subtitle,12,muted)); bar.addView(titles,LinearLayout.LayoutParams(0,-2,1f)); actions.forEach { entry->bar.addView(icon(entry.first,entry.first,entry.second).also { if(entry.first=="more")overflowAnchor=it }) }; root.addView(bar); headerView=bar
    }
    private fun setting(parent:LinearLayout,title:String,description:String,click:()->Unit) {
        val box=column().apply { setPadding(dp(4),dp(16),dp(4),dp(16)) }; box.addView(label(title,17,ink,true)); box.addView(label(description,14,muted).apply { setPadding(0,dp(6),0,0) }); box.setOnClickListener { click() }; when(title) { "Layout"->"layout"; "Permanent Markdown folder"->"markdown"; "Create encrypted backup"->"backup"; "Voice transcription"->"speech"; "Test with 10 notes"->"test-model"; else->null }?.let { box.setTag(R.id.guide_target,it) }; parent.addView(box)
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
