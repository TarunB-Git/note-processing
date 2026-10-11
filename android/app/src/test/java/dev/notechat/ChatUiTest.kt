package dev.notechat

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.text.Spanned
import android.text.style.URLSpan
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode
import java.io.File

@RunWith(RobolectricTestRunner::class)
@Config(sdk=[35],qualifiers="w393dp-h830dp-port-xxhdpi")
@GraphicsMode(GraphicsMode.Mode.NATIVE)
class ChatUiTest {
    private val categories=listOf(Category("thoughts","Thoughts","Observations"),Category("ideas","Ideas","Possibilities"),Category("todos","To-do","Actions"))
    private val messages=listOf(
        Message(id="thought",text="The city feels quieter after the rain.",category="thoughts",created=1791705600000),
        Message(id="idea",text="**Idea:** a little library with a seed exchange.\n\nSee example.com for inspiration.",category="ideas",created=1791705660000),
        Message(id="task",text="Download books for the flight.",category="todos",created=1791705720000),
        Message(id="voice",attachment=Attachment("fixture","Voice message.m4a","audio/mp4",100,28000,listOf(4,18,8,32,11,26,5,21)),created=1791705780000)
    )
    private fun fixture(route:String="all"):Pair<MainActivity,LinearLayout> {
        val activity=Robolectric.buildActivity(MainActivity::class.java).get()
        activity.setTheme(R.style.AppTheme)
        val root=LinearLayout(activity).apply { orientation=LinearLayout.VERTICAL; setBackgroundColor(Color.WHITE) }
        activity.setContentView(root)
        field(activity,"root",root); field(activity,"shown",ChatState(categories,messages)); field(activity,"engine",ChatEngine(ChatState(categories,messages))); field(activity,"route",route); field(activity,"resumed",true)
        call(activity,"render"); layout(root)
        return activity to root
    }
    @Test fun sentBubblesReachTheRightEdgeAndStayWithinPhoneWidth() {
        val (_,root)=fixture()
        val wrappers=descendants(root).filterIsInstance<ReplySwipe>()
        assertTrue(wrappers.isNotEmpty())
        wrappers.forEach { wrapper->
            assertTrue(wrapper.width>root.width*.9)
            val body=wrapper.getChildAt(0)
            assertEquals(wrapper.width-wrapper.paddingRight,body.right)
            assertTrue(body.width<=root.width-root.context.resources.displayMetrics.density*60+1)
        }
        assertFalse(descendants(root).filterIsInstance<TextView>().any { it.text.toString()=="New message" })
        capture(root)
    }
    @Test fun markdownAndBareDomainsBecomeReadableClickableText() {
        val (_,root)=fixture()
        val text=descendants(root).filterIsInstance<MessageText>().first { it.text.contains("little library") }
        assertFalse(text.text.contains("**Idea:**"))
        val links=(text.text as Spanned).getSpans(0,text.text.length,URLSpan::class.java)
        assertTrue(links.any { it.url=="https://example.com" })
    }
    @Test fun longPressSelectsThenTapAddsAndRemovesMessages() {
        val (activity,root)=fixture()
        descendants(root).filterIsInstance<MessageText>().first().performLongClick()
        layout(root)
        assertEquals(setOf("thought"),selected(activity))
        descendants(root).filterIsInstance<MessageText>().first { it.text.contains("Download") }.performClick()
        assertEquals(setOf("thought","task"),selected(activity))
        layout(root)
        descendants(root).filterIsInstance<MessageText>().first { it.text.contains("Download") }.performClick()
        assertEquals(setOf("thought"),selected(activity))
    }
    @Test fun searchKeepsEveryMessageAndNavigatesMatches() {
        val (activity,root)=fixture()
        field(activity,"searchMode",true); field(activity,"query","the"); call(activity,"render"); layout(root)
        assertEquals(4,descendants(root).filterIsInstance<ReplySwipe>().size)
        assertTrue(descendants(root).any { it.contentDescription=="Previous search result" })
        assertTrue(descendants(root).any { it.contentDescription=="Next search result" })
        assertTrue(descendants(root).filterIsInstance<TextView>().any { it.text.toString().endsWith("of 2") })
    }
    @Test fun voiceBubbleHasPlaybackSeekAndSpeedControls() {
        val (_,root)=fixture()
        assertTrue(descendants(root).any { it.contentDescription=="Play voice message" })
        assertTrue(descendants(root).any { it.contentDescription=="Seek voice message" })
        assertTrue(descendants(root).any { it.contentDescription=="Change playback speed" })
        assertTrue(descendants(root).filterIsInstance<TextView>().any { it.text.toString()=="0:28" })
    }
    @Test fun homeHasDiscreetCategoryTotalsAndNoInboxBadge() {
        val (_,root)=fixture("home")
        val text=descendants(root).filterIsInstance<TextView>()
        assertFalse(text.any { it.text.toString()=="4" })
        assertTrue(text.any { it.text.toString()=="1" && it.typeface.isItalic })
        assertTrue(descendants(root).any { it.contentDescription=="Settings" })
    }
    @Test fun holdingThenSlidingLeftCancelsWithoutStartingAnotherRecording() {
        val (activity,_)=fixture()
        val button=RecordButton(activity)
        var starts=0; var cancels=0; var releases=0
        button.startHold={ starts++; true }; button.cancelHold={ cancels++ }; button.releaseHold={ releases++ }; button.setOnClickListener { starts++ }
        val parent=android.widget.FrameLayout(activity); parent.addView(button)
        val density=activity.resources.displayMetrics.density
        val down=android.view.MotionEvent.obtain(0,0,android.view.MotionEvent.ACTION_DOWN,150*density,0f,0)
        button.onTouchEvent(down)
        org.robolectric.Shadows.shadowOf(android.os.Looper.getMainLooper()).idleFor(java.time.Duration.ofMillis(350))
        button.onTouchEvent(android.view.MotionEvent.obtain(0,400,android.view.MotionEvent.ACTION_MOVE,0f,0f,0))
        button.onTouchEvent(android.view.MotionEvent.obtain(0,450,android.view.MotionEvent.ACTION_UP,0f,0f,0))
        assertEquals(1,starts); assertEquals(1,cancels); assertEquals(0,releases)
    }
    @Test fun slidingUpLocksRecordingAndReleaseDoesNotSend() {
        val (activity,_)=fixture()
        val button=RecordButton(activity)
        var locks=0; var sends=0
        button.startHold={ true }; button.lockHold={ locks++; button.gesturesEnabled=false }; button.releaseHold={ sends++ }; button.setOnClickListener { sends++ }
        val parent=android.widget.FrameLayout(activity); parent.addView(button)
        val density=activity.resources.displayMetrics.density
        button.onTouchEvent(android.view.MotionEvent.obtain(0,0,android.view.MotionEvent.ACTION_DOWN,0f,100*density,0))
        org.robolectric.Shadows.shadowOf(android.os.Looper.getMainLooper()).idleFor(java.time.Duration.ofMillis(350))
        button.onTouchEvent(android.view.MotionEvent.obtain(0,400,android.view.MotionEvent.ACTION_MOVE,0f,0f,0))
        button.onTouchEvent(android.view.MotionEvent.obtain(0,450,android.view.MotionEvent.ACTION_UP,0f,0f,0))
        assertEquals(1,locks); assertEquals(0,sends)
    }
    @Test fun swipeReplyWorksInEitherDirection() {
        for(direction in listOf(-1,1)) {
        val (activity,root)=fixture()
        val wrapper=descendants(root).filterIsInstance<ReplySwipe>().first()
        val density=activity.resources.displayMetrics.density
        wrapper.onInterceptTouchEvent(android.view.MotionEvent.obtain(0,0,android.view.MotionEvent.ACTION_DOWN,100*density,0f,0))
        assertTrue(wrapper.onInterceptTouchEvent(android.view.MotionEvent.obtain(0,50,android.view.MotionEvent.ACTION_MOVE,(100+direction*80)*density,0f,0)))
        wrapper.onTouchEvent(android.view.MotionEvent.obtain(0,60,android.view.MotionEvent.ACTION_MOVE,(100+direction*80)*density,0f,0))
        wrapper.onTouchEvent(android.view.MotionEvent.obtain(0,70,android.view.MotionEvent.ACTION_UP,(100+direction*80)*density,0f,0))
        assertEquals("thought",MainActivity::class.java.getDeclaredField("reply").apply { isAccessible=true }.get(activity))
        }
    }
    @Test fun modelDashboardShowsBuiltInLearnerAndOptionalEncoder() {
        val (_,root)=fixture("model")
        val text=descendants(root).filterIsInstance<TextView>().map { it.text.toString() }
        assertTrue(text.contains("Local learner · installed"))
        assertTrue(text.any { it.contains("Optional download") })
        assertTrue(descendants(root).any { it is ModelFlowView })
        capture(root,"model")
    }
    @Test fun lockedRecordingComposerShowsPauseDiscardTimerAndSend() {
        val (activity,root)=fixture()
        field(activity,"recordingLocked",true); field(activity,"recordingStarted",System.currentTimeMillis()-12000)
        call(activity,"updateRecordingComposer"); layout(root)
        assertTrue(descendants(root).any { it.contentDescription=="Pause recording" })
        assertTrue(descendants(root).any { it.contentDescription=="Discard recording" })
        assertTrue(descendants(root).filterIsInstance<TextView>().any { it.text.toString()=="0:12" })
        capture(root,"recording")
    }
    @Test fun homeHasNoTaglineAndPlacesUnsortedAfterCategories() {
        val (_,root)=fixture("home")
        val labels=descendants(root).filterIsInstance<TextView>().map { it.text.toString() }
        assertFalse(labels.contains("Chats")); assertFalse(labels.any { it.contains("Everything stays") })
        assertTrue(labels.indexOf("Unsorted")>labels.indexOf("To-do"))
    }
    @Test fun todoChecklistHasLargeLeftCheckboxesAndNoChatDateChips() {
        val (_,root)=fixture("todos")
        val check=descendants(root).filterIsInstance<android.widget.CheckBox>().single()
        assertTrue(check.width>=52*root.resources.displayMetrics.density)
        assertTrue(check.height>=52*root.resources.displayMetrics.density)
        assertEquals(0,(check.parent as ViewGroup).indexOfChild(check))
        assertFalse(descendants(root).any { it is ReplySwipe })
        capture(root,"todo")
    }
    @Test fun darkModeChangesSurfacesAndKeepsTextReadable() {
        val (activity,root)=fixture()
        field(activity,"shown",ChatState(categories,messages,preferences=Preferences(appearance="dark")))
        call(activity,"render"); layout(root)
        val text=descendants(root).filterIsInstance<MessageText>().first()
        assertTrue(Color.red(text.currentTextColor)>200)
        assertTrue(UiColors.of(activity).dark)
        capture(root,"dark")
    }
    @Test fun splitOpensInNormalComposerWithSendActionsForEachLine() {
        val (activity,root)=fixture()
        val message=Message(id="thought",text="First fragment\nSecond fragment",created=123)
        MainActivity::class.java.getDeclaredMethod("startSplit",Message::class.java).apply { isAccessible=true }.invoke(activity,message)
        layout(root)
        assertEquals(message.text,descendants(root).filterIsInstance<android.widget.EditText>().last().text.toString())
        assertTrue(descendants(root).any { it.contentDescription=="Send through line 1" })
        assertTrue(descendants(root).any { it.contentDescription=="Send through line 2" })
        capture(root,"split")
    }
    @Test fun unsortedIncludesSuggestedAndManualTagsUntilApproved() {
        val (activity,root)=fixture("unsorted")
        val pending=messages.map { if(it.id=="thought")it.copy(approved=true) else if(it.id=="task")it.copy(manual=true) else it }
        field(activity,"shown",ChatState(categories,pending)); call(activity,"render"); layout(root)
        val text=descendants(root).filterIsInstance<MessageText>().map { it.text.toString() }
        assertFalse(text.any { it.contains("city feels") })
        assertTrue(text.any { it.contains("little library") })
        assertTrue(text.any { it.contains("Download books") })
        assertTrue(descendants(root).any { it is VoiceMessageView })
        val engine=ChatEngine(ChatState(categories,pending))
        engine.approve("idea")
        field(activity,"shown",engine.state); call(activity,"render"); layout(root)
        assertFalse(descendants(root).filterIsInstance<MessageText>().any { it.text.contains("little library") })
    }
    @Test fun deletedMessagesUseChatBubblesWithFullTextAndVoiceControls() {
        val (activity,root)=fixture("deleted")
        val now=System.currentTimeMillis()
        val rows=listOf(
            messages[1].copy(text="**Saved note**\n\n"+(1..12).joinToString("\n") { "Line $it" },deleted=now),
            messages.last().copy(deleted=now-1000),
            messages[0].copy(deleted=now-RETENTION_MS)
        )
        field(activity,"shown",ChatState(categories,rows)); call(activity,"render"); layout(root)
        val text=descendants(root).filterIsInstance<MessageText>().single()
        assertTrue(text.text.contains("Line 12")); assertFalse(text.text.contains("**"))
        assertTrue(text.maxLines>3)
        assertFalse(text.performLongClick())
        assertEquals(2,descendants(root).filterIsInstance<ReplySwipe>().size)
        assertTrue(descendants(root).any { it.contentDescription=="Play voice message" })
        assertEquals(2,descendants(root).filterIsInstance<TextView>().count { it.text.toString()=="Restore" })
        assertFalse(descendants(root).any { it.contentDescription?.toString()?.startsWith("Approve category")==true })
        assertFalse(descendants(root).any { it is android.widget.EditText })
        capture(root,"deleted")
    }
    @Test fun deletedImageKeepsItsPreviewInsideTheMessageBubble() {
        val (activity,root)=fixture("deleted")
        val image=Message(id="photo",attachment=Attachment("photo-fixture","Photo.png","image/png",100),category="thoughts",deleted=System.currentTimeMillis())
        val bitmap=Bitmap.createBitmap(240,180,Bitmap.Config.ARGB_8888).apply { eraseColor(Color.rgb(85,141,118)) }
        @Suppress("UNCHECKED_CAST")
        val cache=MainActivity::class.java.getDeclaredField("imageCache").apply { isAccessible=true }.get(activity) as android.util.LruCache<String,Bitmap>
        cache.put("photo-fixture",bitmap)
        field(activity,"shown",ChatState(categories,listOf(image))); call(activity,"render"); layout(root)
        val preview=descendants(root).filterIsInstance<android.widget.ImageView>().single { it.contentDescription=="Image: Photo.png" }
        assertSame(bitmap,(preview.drawable as android.graphics.drawable.BitmapDrawable).bitmap)
        assertFalse(preview.performLongClick())
        assertTrue(preview.width>0 && preview.width<root.width)
        capture(root,"deleted-image")
    }
    @Test fun manuallyTaggedMediaCanBeApprovedWithoutTeachingTheTextModel() {
        val (activity,root)=fixture()
        val media=messages.last().copy(category="thoughts",manual=true)
        field(activity,"shown",ChatState(categories,listOf(media))); call(activity,"render"); layout(root)
        assertTrue(descendants(root).any { it.contentDescription=="Approve category" })
        val engine=ChatEngine(ChatState(categories,listOf(media)))
        engine.approve(media.id)
        assertTrue(engine.state.messages.single().approved)
        assertTrue(engine.confirmed().isEmpty())
    }
    @Test fun completingTasksAndRebuildingKeepsTheVisibleMessageAndOffset() {
        val (activity,root)=fixture("todos")
        val tasks=(0..59).map { Message(id="task-$it",text="Task $it",category="todos") }
        val engine=ChatEngine(ChatState(categories,tasks))
        field(activity,"shown",engine.state); call(activity,"render"); layout(root)
        var recycler=descendants(root).filterIsInstance<androidx.recyclerview.widget.RecyclerView>().single()
        var manager=recycler.layoutManager as androidx.recyclerview.widget.LinearLayoutManager
        manager.scrollToPositionWithOffset(25,-18); layout(root)
        val before=manager.findFirstVisibleItemPosition()
        val offset=manager.findViewByPosition(before)!!.top
        engine.complete("task-25",true)
        field(activity,"shown",engine.state); call(activity,"render"); layout(root)
        recycler=descendants(root).filterIsInstance<androidx.recyclerview.widget.RecyclerView>().single()
        manager=recycler.layoutManager as androidx.recyclerview.widget.LinearLayoutManager
        assertEquals(before,manager.findFirstVisibleItemPosition())
        assertEquals(offset,manager.findViewByPosition(before)!!.top)
        val completed=descendants(root).filterIsInstance<MessageText>().first { it.text.toString()=="Task 25" }
        assertTrue(completed.paintFlags and android.graphics.Paint.STRIKE_THRU_TEXT_FLAG != 0)
    }
    @Test fun reopenedWorkspaceRestoresTheCategoryAndMessageAnchor() {
        val (activity,root)=fixture()
        val tasks=(0..59).map { Message(id="task-$it",text="Task $it",category="todos") }
        val state=ChatState(categories,tasks,navigation=Navigation("todos","task-25",25,-18))
        field(activity,"route","home"); field(activity,"renderedRoute",""); MainActivity::class.java.getDeclaredField("list").apply { isAccessible=true }.set(activity,null)
        field(activity,"shown",state)
        MainActivity::class.java.getDeclaredMethod("restoreLocation",ChatState::class.java).apply { isAccessible=true }.invoke(activity,state)
        call(activity,"render"); layout(root)
        val manager=descendants(root).filterIsInstance<androidx.recyclerview.widget.RecyclerView>().single().layoutManager as androidx.recyclerview.widget.LinearLayoutManager
        assertEquals(25,manager.findFirstVisibleItemPosition())
        assertEquals(-18,manager.findViewByPosition(25)!!.top)
        assertTrue(descendants(root).filterIsInstance<TextView>().any { it.text.toString()=="To-do" })
    }
    @Test fun replyPreviewIsCompactAndTappingItHighlightsTheOriginal() {
        val (activity,root)=fixture()
        val state=ChatState(categories,messages+Message(id="reply",text="Worth keeping",replyTo="thought"))
        field(activity,"shown",state); call(activity,"render"); layout(root)
        val preview=descendants(root).filterIsInstance<ReplyPreview>().single()
        assertEquals(2,preview.summary.maxLines)
        assertEquals(messages.first().text,preview.summary.text.toString())
        preview.performClick(); layout(root)
        org.robolectric.Shadows.shadowOf(android.os.Looper.getMainLooper()).idle()
        layout(root)
        val target=descendants(root).first { it.tag=="thought" }
        assertNotNull(target.foreground)
        capture(root,"reply")
    }
    @Test fun scrollingAboveLatestShowsTheReturnButton() {
        val (activity,root)=fixture()
        val rows=(0..59).map { Message(id="note-$it",text="Note $it") }
        field(activity,"shown",ChatState(categories,rows)); call(activity,"render"); layout(root)
        val viewport=descendants(root).filterIsInstance<ChatViewport>().single()
        val manager=viewport.recycler.layoutManager as androidx.recyclerview.widget.LinearLayoutManager
        manager.scrollToPositionWithOffset(10,0); layout(root); viewport.updateLatest()
        assertEquals(View.VISIBLE,viewport.latest.visibility)
        capture(root,"scroll")
        assertTrue(viewport.latest.performClick())
        org.robolectric.Shadows.shadowOf(android.os.Looper.getMainLooper()).idle()
        layout(root)
        assertTrue(manager.findFirstVisibleItemPosition()>20)
    }
    @Test fun taskTextRowAndCheckboxAllToggleCompletionWhileLongPressSelects() {
        val (activity,_)=fixture("todos")
        val engine=ChatEngine(ChatState(categories,listOf(messages[2])))
        var selections=0
        fun task()=TodoRow(activity,engine.state.messages.single(),UiColors(false),MarkdownText(activity,{}),
            { engine.complete("task",!engine.state.messages.single().done) },{ selections++ })
        task().text.performClick(); assertTrue(engine.state.messages.single().done)
        assertTrue(task().text.paintFlags and android.graphics.Paint.STRIKE_THRU_TEXT_FLAG != 0)
        task().performClick(); assertFalse(engine.state.messages.single().done)
        task().check.performClick(); assertTrue(engine.state.messages.single().done)
        task().text.performLongClick(); assertEquals(1,selections)
        assertTrue(engine.state.messages.single().done)
    }
    private fun capture(root:LinearLayout,name:String="inbox") {
        val bitmap=Bitmap.createBitmap(root.width,root.height,Bitmap.Config.ARGB_8888)
        root.draw(Canvas(bitmap))
        val directory=File("build/reports/chat-ui").apply { mkdirs() }
        File(directory,"$name.png").outputStream().use { bitmap.compress(Bitmap.CompressFormat.PNG,100,it) }
        assertTrue(bitmap.width>0)
    }
    private fun layout(root:View) { val density=root.resources.displayMetrics.density; val width=(393*density).toInt(); val height=(830*density).toInt(); root.measure(View.MeasureSpec.makeMeasureSpec(width,View.MeasureSpec.EXACTLY),View.MeasureSpec.makeMeasureSpec(height,View.MeasureSpec.EXACTLY)); root.layout(0,0,width,height) }
    private fun descendants(view:View):List<View> = listOf(view)+(if(view is ViewGroup)(0 until view.childCount).flatMap { descendants(view.getChildAt(it)) } else emptyList())
    private fun field(activity:MainActivity,name:String,value:Any) { MainActivity::class.java.getDeclaredField(name).apply { isAccessible=true }.set(activity,value) }
    private fun call(activity:MainActivity,name:String) { MainActivity::class.java.getDeclaredMethod(name).apply { isAccessible=true }.invoke(activity) }
    @Suppress("UNCHECKED_CAST") private fun selected(activity:MainActivity)=MainActivity::class.java.getDeclaredField("selected").apply { isAccessible=true }.get(activity) as Set<String>
}
