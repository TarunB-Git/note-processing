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
        field(activity,"persistState",{ _:ChatState -> Unit }); field(activity,"root",root); field(activity,"shown",ChatState(categories,messages)); field(activity,"engine",ChatEngine(ChatState(categories,messages))); field(activity,"route",route); field(activity,"resumed",true)
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
            assertTrue(body.width<=wrapper.width-wrapper.paddingLeft-wrapper.paddingRight)
            val bubble=descendants(body).filterIsInstance<BubbleLayout>().single()
            assertTrue(bubble.width<=root.width-root.context.resources.displayMetrics.density*60+1)
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
        listOf("Notes tab","Chats tab").forEach { tab ->
            val icon=descendants(descendants(root).first { it.contentDescription==tab }).filterIsInstance<android.widget.ImageView>().single()
            assertEquals(icon.drawable.bounds.width(),icon.drawable.bounds.height())
        }
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
        assertEquals(1,labels.count { it=="Chats" }); assertFalse(labels.any { it.contains("Everything stays") })
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
    private fun capture(root:View,name:String="inbox") {
        org.robolectric.Shadows.shadowOf(android.os.Looper.getMainLooper()).idleFor(java.time.Duration.ofMillis(250)); layout(root)
        val bitmap=Bitmap.createBitmap(root.width,root.height,Bitmap.Config.ARGB_8888)
        root.draw(Canvas(bitmap))
        val directory=File("build/reports/chat-ui").apply { mkdirs() }
        File(directory,"$name.png").outputStream().use { bitmap.compress(Bitmap.CompressFormat.PNG,100,it) }
        assertTrue(bitmap.width>0)
    }
    @Test fun selectingFromTheEmptySideKeepsTheExactViewportAnchor() {
        val (activity,root)=fixture()
        val rows=(0..59).map { Message(id="row-$it",text="Note $it",category="thoughts") }
        field(activity,"shown",ChatState(categories,rows)); call(activity,"render"); layout(root)
        val list=descendants(root).filterIsInstance<androidx.recyclerview.widget.RecyclerView>().single()
        val manager=list.layoutManager as androidx.recyclerview.widget.LinearLayoutManager
        manager.scrollToPositionWithOffset(25,-15); layout(root)
        val before=manager.findViewByPosition(25)!!.top
        descendants(root).filterIsInstance<ReplySwipe>().first().performLongClick(); layout(root)
        val changed=descendants(root).filterIsInstance<androidx.recyclerview.widget.RecyclerView>().single().layoutManager as androidx.recyclerview.widget.LinearLayoutManager
        assertEquals(25,changed.findFirstVisibleItemPosition())
        assertEquals(before,changed.findViewByPosition(25)!!.top)
        assertSame(list,descendants(root).filterIsInstance<androidx.recyclerview.widget.RecyclerView>().single())
        descendants(root).filterIsInstance<ReplySwipe>()[1].performClick(); layout(root)
        assertEquals(2,selected(activity).size)
        call(activity,"clearSelection"); layout(root)
        assertTrue(selected(activity).isEmpty())
        assertSame(list,descendants(root).filterIsInstance<androidx.recyclerview.widget.RecyclerView>().single())
        assertEquals(before,manager.findViewByPosition(25)!!.top)
    }
    @Test fun appendingAtBottomRevealsTheNewMessageAndDoesNotOverrideScrolledUpPosition() {
        val (activity,root)=fixture()
        val rows=(0..59).map { Message(id="row-$it",text="Note $it") }
        field(activity,"shown",ChatState(categories,rows)); field(activity,"followLatest",true); call(activity,"render"); layout(root)
        val newer=rows+Message(id="sent",text="Newly sent message")
        field(activity,"shown",ChatState(categories,newer)); field(activity,"followLatest",true); call(activity,"render"); layout(root)
        org.robolectric.Shadows.shadowOf(android.os.Looper.getMainLooper()).idle(); layout(root)
        var manager=descendants(root).filterIsInstance<androidx.recyclerview.widget.RecyclerView>().single().layoutManager as androidx.recyclerview.widget.LinearLayoutManager
        assertEquals(60,manager.findLastCompletelyVisibleItemPosition())
        manager.scrollToPositionWithOffset(20,-10); layout(root)
        field(activity,"shown",ChatState(categories,newer+Message(id="later",text="Another message"))); call(activity,"render"); layout(root)
        manager=descendants(root).filterIsInstance<androidx.recyclerview.widget.RecyclerView>().single().layoutManager as androidx.recyclerview.widget.LinearLayoutManager
        assertEquals(20,manager.findFirstVisibleItemPosition())
        assertEquals(-10,manager.findViewByPosition(20)!!.top)
    }
    @Test fun pinnedBannerJumpsToAndHighlightsThePinnedMessage() {
        val (activity,root)=fixture()
        val rows=(0..59).map { Message(id="row-$it",text="Note $it",pinnedUntil=if(it==5)System.currentTimeMillis()+ARCHIVE_DELAY_MS else null) }
        field(activity,"shown",ChatState(categories,rows)); call(activity,"render"); layout(root)
        val bar=descendants(root).first { it.contentDescription=="Pinned message. Tap to jump." }
        bar.performClick(); layout(root)
        org.robolectric.Shadows.shadowOf(android.os.Looper.getMainLooper()).idle(); layout(root)
        assertNotNull(descendants(root).first { it.tag=="row-5" }.foreground)
        capture(root,"pinned")
    }
    @Test fun wikiLinksRenderAsLocalLinksAndCodeStaysLiteral() {
        val (activity,_)=fixture()
        val text=MessageText(activity) {}
        MarkdownText(activity,{}).render(text,"See [[Thoughts|my notes]]. Keep `[[literal]]`.","",{})
        assertTrue(text.text.contains("my notes"))
        assertTrue(text.text.contains("[[literal]]"))
        assertTrue((text.text as Spanned).getSpans(0,text.text.length,URLSpan::class.java).any { it.url=="notechat-md:Thoughts" })
    }

    @Test fun completionChangesTheRowBeforePersistenceAndDoesNotWriteMarkdown() {
        val (activity,root)=fixture("todos")
        val entered=java.util.concurrent.CountDownLatch(1)
        val release=java.util.concurrent.CountDownLatch(1)
        field(activity,"persistState",{ _:ChatState -> entered.countDown(); release.await(5,java.util.concurrent.TimeUnit.SECONDS); Unit })
        val recycler=descendants(root).filterIsInstance<androidx.recyclerview.widget.RecyclerView>().single()
        descendants(root).filterIsInstance<TodoRow>().single().performClick()
        layout(root)
        assertTrue(descendants(root).filterIsInstance<TodoRow>().single().check.isChecked)
        assertSame(recycler,descendants(root).filterIsInstance<androidx.recyclerview.widget.RecyclerView>().single())
        assertTrue(entered.await(5,java.util.concurrent.TimeUnit.SECONDS))
        assertFalse(MainActivity::class.java.getDeclaredField("markdownDirty").apply { isAccessible=true }.getBoolean(activity))
        release.countDown()
    }
    @Test fun boardsAndGroupedSettingsHaveNativeControls() {
        val (activity,root)=fixture("home")
        field(activity,"shown",ChatState(categories,messages,preferences=Preferences(frontend="notes"),chats=listOf(Chat("journal","Journal"))))
        call(activity,"render"); layout(root)
        assertTrue(descendants(root).any { it is NoteBoard }); capture(root,"boards")
        field(activity,"route","settings"); call(activity,"render"); layout(root)
        val labels=descendants(root).filterIsInstance<TextView>().map { it.text.toString() }
        assertTrue(labels.contains("Storage & backups")); assertTrue(labels.contains("MODs")); assertTrue(labels.contains("User guide")); capture(root,"settings")
    }

    @Test fun taskTouchCompletesOnReleaseWithoutTheDoubleTapTimeout() {
        val (activity,_)=fixture()
        var complete=false
        val task=TodoRow(activity,messages[2],UiColors(false),MarkdownText(activity,{}),{ complete=!complete },{})
        task.text.onTouchEvent(android.view.MotionEvent.obtain(0,0,android.view.MotionEvent.ACTION_DOWN,10f,10f,0))
        task.text.onTouchEvent(android.view.MotionEvent.obtain(0,50,android.view.MotionEvent.ACTION_UP,10f,10f,0))
        assertTrue(complete)
    }

    @Test fun markdownWaitsForThirtySecondsAfterTheLastInteractionAndRetriesFailure() {
        val (activity,root)=fixture("todos")
        val state=ChatState(categories,messages,markdown=MarkdownState("content://test/folder"))
        field(activity,"shown",state); field(activity,"engine",ChatEngine(state))
        field(activity,"persistState",{ _:ChatState -> Unit })
        val writes=java.util.concurrent.atomic.AtomicInteger()
        field(activity,"exportFolder",{ _:ChatEngine -> if(writes.incrementAndGet()==1)error("Folder temporarily unavailable"); 0 })
        val worker=MainActivity::class.java.getDeclaredField("worker").apply { isAccessible=true }.get(activity) as java.util.concurrent.ExecutorService
        descendants(root).filterIsInstance<TodoRow>().single().performClick()
        worker.submit {}.get(5,java.util.concurrent.TimeUnit.SECONDS)
        val looper=org.robolectric.Shadows.shadowOf(android.os.Looper.getMainLooper())
        looper.idleFor(java.time.Duration.ofSeconds(29)); assertEquals(0,writes.get())
        activity.onUserInteraction()
        looper.idleFor(java.time.Duration.ofSeconds(29)); assertEquals(0,writes.get())
        looper.idleFor(java.time.Duration.ofSeconds(1)); worker.submit {}.get(5,java.util.concurrent.TimeUnit.SECONDS); looper.idle()
        assertEquals(1,writes.get())
        assertTrue(MainActivity::class.java.getDeclaredField("markdownDirty").apply { isAccessible=true }.getBoolean(activity))
        looper.idleFor(java.time.Duration.ofSeconds(30)); worker.submit {}.get(5,java.util.concurrent.TimeUnit.SECONDS); looper.idle()
        assertEquals(2,writes.get())
        assertFalse(MainActivity::class.java.getDeclaredField("markdownDirty").apply { isAccessible=true }.getBoolean(activity))
    }

    @Test fun aRejectedTagCannotBeApprovedWhileItsReplacementIsPending() {
        val (activity,root)=fixture()
        val entered=java.util.concurrent.CountDownLatch(1)
        val release=java.util.concurrent.CountDownLatch(1)
        field(activity,"persistState",{ _:ChatState -> entered.countDown(); release.await(5,java.util.concurrent.TimeUnit.SECONDS); Unit })
        MainActivity::class.java.getDeclaredMethod("disapprove",Message::class.java).apply { isAccessible=true }.invoke(activity,messages.first())
        layout(root)
        val state=MainActivity::class.java.getDeclaredField("shown").apply { isAccessible=true }.get(activity) as ChatState
        assertNull(state.messages.first().category)
        assertThrows(IllegalArgumentException::class.java) { ChatEngine(state).approve(messages.first().id) }
        assertTrue(entered.await(5,java.util.concurrent.TimeUnit.SECONDS)); release.countDown()
    }

    @Test fun restoringWeightsWithoutTheirOptionalEncoderPreservesThemForLater() {
        val (activity,_)=fixture()
        val head=PersonalHead("different-encoder-signature",categories.map { List(524) { 0f } },List(categories.size) { 0f })
        val engine=ChatEngine(ChatState(categories,messages,model=head))
        field(activity,"engine",engine)
        call(activity,"configure")
        assertEquals(head,engine.state.model)
        val report=org.json.JSONObject(ModelReport.encode(engine.state,null))
        assertEquals("starter",report.getString("activeHead"))
        assertTrue(report.getBoolean("storedPersonalHead"))
    }

    @Test fun notesHomeHasOneWideInboxAndSeparateTwoColumnTablesWithScrollablePreviews() {
        val (activity,root)=fixture("home")
        val notes=(0..4).map { Message(id="note-$it",text="Thought $it",category="thoughts") }
        val state=ChatState(categories,notes+Message(id="private",text="Private journal entry",chatId="journal"),preferences=Preferences(frontend="notes"),chats=listOf(Chat("journal","Journal"),Chat("reading","Reading")))
        field(activity,"shown",state); call(activity,"render"); layout(root)
        val board=descendants(root).filterIsInstance<NoteBoard>().single()
        val grid=board.layoutManager as androidx.recyclerview.widget.GridLayoutManager
        assertEquals(2,grid.spanCount)
        assertEquals(2,grid.spanSizeLookup.getSpanSize(0))
        assertEquals(1,grid.spanSizeLookup.getSpanSize(2)); assertEquals(1,grid.spanSizeLookup.getSpanSize(3))
        assertEquals(2,grid.spanSizeLookup.getSpanSize(6)); assertEquals(1,grid.spanSizeLookup.getSpanSize(7))
        val cards=descendants(root).filter { it.contentDescription?.startsWith("Open ")==true }
        assertEquals(1,cards.count { it.contentDescription=="Open Inbox board" })
        assertTrue(cards.any { it.contentDescription=="Open Thoughts board" })
        val inbox=cards.first { it.contentDescription=="Open Inbox board" }
        assertFalse(descendants(inbox).filterIsInstance<MessageText>().any { it.text.contains("Private journal") })
        assertTrue(descendants(root).filterIsInstance<androidx.recyclerview.widget.RecyclerView>().any { (it.layoutManager as? androidx.recyclerview.widget.LinearLayoutManager)?.orientation==androidx.recyclerview.widget.RecyclerView.HORIZONTAL })
        capture(root,"notes-home")
        grid.scrollToPosition(7); layout(root)
        assertTrue(descendants(root).any { it.contentDescription=="Open Journal board" })
        assertTrue(descendants(root).any { it.contentDescription=="Open Reading board" })
        capture(root,"notes-private-boards")
    }
    @Test fun privateInboxOptInAlsoControlsSearchAndPrivateResultsOpenTheirOwnChat() {
        val (activity,root)=fixture()
        val private=Message(id="private",text="secret-from-chat",chatId="journal")
        val state=ChatState(categories,messages+private,chats=listOf(Chat("journal","Journal")))
        field(activity,"shown",state); field(activity,"engine",ChatEngine(state)); call(activity,"render"); layout(root)
        assertFalse(descendants(root).filterIsInstance<MessageText>().any { it.text.contains(private.text) })
        field(activity,"shown",state.copy(preferences=Preferences(includePrivateChatsInInbox=true))); call(activity,"render"); layout(root)
        assertTrue(descendants(root).filterIsInstance<MessageText>().any { it.text.contains(private.text) })
        field(activity,"shown",state); field(activity,"route","home"); field(activity,"query",private.text); call(activity,"render"); layout(root)
        assertEquals(0,descendants(root).filterIsInstance<androidx.recyclerview.widget.RecyclerView>().single().adapter!!.itemCount)
        field(activity,"route","chats"); call(activity,"render"); layout(root)
        val result=descendants(root).first { it.tag=="message:private" } as ViewGroup
        result.getChildAt(0).performClick(); layout(root)
        assertEquals("chat:journal",MainActivity::class.java.getDeclaredField("route").apply { isAccessible=true }.get(activity))
        assertTrue(descendants(root).filterIsInstance<MessageText>().any { it.text.contains(private.text) })
        MainActivity::class.java.getDeclaredMethod("startSplit",Message::class.java).apply { isAccessible=true }.invoke(activity,private)
        layout(root)
        assertEquals("chat:journal",MainActivity::class.java.getDeclaredField("route").apply { isAccessible=true }.get(activity))
        assertEquals(private.text,descendants(root).filterIsInstance<android.widget.EditText>().single().text.toString())
    }
    @Test fun privateChatFormKeepsDuplicateNamesOpenForCorrection() {
        val (activity,_)=fixture("chats")
        val state=ChatState(categories,chats=listOf(Chat("journal","Journal")))
        field(activity,"shown",state); field(activity,"engine",ChatEngine(state)); call(activity,"newChat")
        val dialogs=MainActivity::class.java.getDeclaredField("dialogs").apply { isAccessible=true }.get(activity) as List<*>
        val sheet=dialogs.single() as com.google.android.material.bottomsheet.BottomSheetDialog
        val form=descendants(sheet.window!!.decorView).first { it.tag=="chat-name-form" }
        val name=descendants(form).filterIsInstance<com.google.android.material.textfield.TextInputEditText>().single()
        val submit=descendants(form).first { it.contentDescription=="Create chat" }
        assertFalse(submit.isEnabled)
        name.setText("Journal"); submit.performClick()
        val field=descendants(form).filterIsInstance<com.google.android.material.textfield.TextInputLayout>().single()
        assertEquals("That chat name is already used",field.error.toString()); assertTrue(sheet.isShowing)
        name.setText("Reading"); assertNull(field.error); assertTrue(submit.isEnabled)
        capture(sheet.window!!.decorView,"new-private-chat")
        descendants(form).first { it.contentDescription=="Cancel" }.performClick(); assertFalse(sheet.isShowing)
    }
    @Test fun chatsTabOnlyShowsPrivateChatsAndReturnsWithItsOwnScrollPosition() {
        val (activity,root)=fixture("home")
        val state=ChatState(categories,messages,chats=(0..19).map { Chat("private-$it","Notebook $it") })
        field(activity,"shown",state); field(activity,"engine",ChatEngine(state)); call(activity,"render"); layout(root)
        assertFalse(descendants(root).filterIsInstance<TextView>().any { it.text.toString()=="Notebook 0" })
        descendants(root).first { it.contentDescription=="Chats tab" }.performClick(); layout(root)
        assertTrue(descendants(root).filterIsInstance<TextView>().any { it.text.toString()=="Notebook 0" })
        assertFalse(descendants(root).filterIsInstance<TextView>().any { it.text.toString()=="Thoughts" })
        var list=descendants(root).filterIsInstance<androidx.recyclerview.widget.RecyclerView>().single()
        (list.layoutManager as androidx.recyclerview.widget.LinearLayoutManager).scrollToPositionWithOffset(10,-12); layout(root)
        descendants(root).first { it.contentDescription=="Notes tab" }.performClick(); layout(root)
        descendants(root).first { it.contentDescription=="Chats tab" }.performClick(); layout(root)
        list=descendants(root).filterIsInstance<androidx.recyclerview.widget.RecyclerView>().single()
        assertEquals(10,(list.layoutManager as androidx.recyclerview.widget.LinearLayoutManager).findFirstVisibleItemPosition())
        capture(root,"private-tab")
    }
    @Test fun notesOpenAsADocumentAndEditInlineWithoutLeavingTheCategory() {
        val (activity,root)=fixture("thoughts")
        val state=ChatState(categories,messages,preferences=Preferences(frontend="notes"))
        field(activity,"shown",state); field(activity,"engine",ChatEngine(state)); call(activity,"render"); layout(root)
        assertFalse(descendants(root).any { it is ReplySwipe })
        val text=descendants(root).filterIsInstance<MessageText>().first { it.text.contains("city feels") }
        text.performClick(); layout(root)
        assertEquals("thoughts",MainActivity::class.java.getDeclaredField("route").apply { isAccessible=true }.get(activity))
        val editor=descendants(root).filterIsInstance<android.widget.EditText>().single()
        assertEquals(messages.first().text,editor.text.toString())
        assertTrue(descendants(root).filterIsInstance<TextView>().any { it.text.toString()=="Editing note" })
        capture(root,"notes-editor")
        editor.setText("A revised observation")
        call(activity,"cancelEdit"); layout(root)
        assertTrue(descendants(root).filterIsInstance<MessageText>().any { it.text.toString()==messages.first().text })
    }
    @Test fun homeSwipeSwitchesTabsButLeavesBoardPreviewAndVerticalGesturesAlone() {
        val (activity,root)=fixture("home")
        val state=ChatState(categories,messages,preferences=Preferences(frontend="notes"),chats=listOf(Chat("journal","Journal")))
        field(activity,"shown",state); field(activity,"engine",ChatEngine(state)); call(activity,"render"); layout(root)
        layout(activity.window.decorView)
        org.robolectric.Shadows.shadowOf(android.os.Looper.getMainLooper()).idleFor(java.time.Duration.ofMillis(250))
        layout(root)
        val swipe=descendants(root).filterIsInstance<HomeSwipe>().single()
        val preview=descendants(root).filterIsInstance<androidx.recyclerview.widget.RecyclerView>().first { (it.layoutManager as? androidx.recyclerview.widget.LinearLayoutManager)?.orientation==androidx.recyclerview.widget.RecyclerView.HORIZONTAL }
        val bounds=android.graphics.Rect(); assertTrue(preview.getGlobalVisibleRect(bounds))
        val density=root.resources.displayMetrics.density
        fun event(action:Int,x:Float,y:Float)=android.view.MotionEvent.obtain(0,10,action,x,y,0)
        val x=bounds.centerX().toFloat(); val y=bounds.centerY().toFloat()
        event(android.view.MotionEvent.ACTION_DOWN,x,y).also { assertFalse(swipe.onInterceptTouchEvent(it)); it.recycle() }
        event(android.view.MotionEvent.ACTION_MOVE,x-100*density,y).also { assertFalse(swipe.onInterceptTouchEvent(it)); it.recycle() }
        event(android.view.MotionEvent.ACTION_DOWN,300*density,1f).also { assertFalse(swipe.onInterceptTouchEvent(it)); it.recycle() }
        event(android.view.MotionEvent.ACTION_MOVE,200*density,100*density).also { assertFalse(swipe.onInterceptTouchEvent(it)); it.recycle() }
        event(android.view.MotionEvent.ACTION_DOWN,300*density,1f).also { swipe.onInterceptTouchEvent(it); it.recycle() }
        event(android.view.MotionEvent.ACTION_MOVE,190*density,1f).also { assertTrue(swipe.onInterceptTouchEvent(it)); it.recycle() }
        event(android.view.MotionEvent.ACTION_UP,190*density,1f).also { assertTrue(swipe.onTouchEvent(it)); it.recycle() }
        layout(root)
        assertTrue(descendants(root).any { it.contentDescription=="Open Journal board" })
        assertFalse(descendants(root).any { it.contentDescription=="Open Inbox board" })
    }
    @Test fun guideTravelsAcrossRealScreensAndRestoresItsStepDraftAndCaret() {
        val (activity,root)=fixture()
        val input=descendants(root).filterIsInstance<android.widget.EditText>().single()
        input.setText("An unfinished note"); input.setSelection(4)
        MainActivity::class.java.getDeclaredMethod("navigate",String::class.java).apply { isAccessible=true }.invoke(activity,"settings")
        MainActivity::class.java.getDeclaredMethod("startGuide",Boolean::class.javaPrimitiveType,Int::class.javaObjectType).apply { isAccessible=true }.invoke(activity,true,0)
        layout(root)
        val looper=org.robolectric.Shadows.shadowOf(android.os.Looper.getMainLooper()); looper.idleFor(java.time.Duration.ofMillis(250)); layout(root)
        assertTrue(descendants(root).any { it is GuideCoach })
        assertEquals("An unfinished note",descendants(root).filterIsInstance<android.widget.EditText>().single().text.toString())
        assertEquals(4,descendants(root).filterIsInstance<android.widget.EditText>().single().selectionEnd)
        assertNotNull(descendants(root).first { it.getTag(R.id.guide_target)=="composer" }.foreground)
        repeat(4) { descendants(root).first { it.contentDescription=="Guide Next" }.performClick(); layout(root) }
        assertTrue(descendants(root).any { it.contentDescription=="New private chat" })
        assertTrue(descendants(root).any { it.contentDescription?.startsWith("Guide step 5:")==true })
        capture(root,"guided-private-tab")
        MainActivity::class.java.getDeclaredMethod("saveSession").apply { isAccessible=true }.invoke(activity)
        val worker=MainActivity::class.java.getDeclaredField("worker").apply { isAccessible=true }.get(activity) as java.util.concurrent.ExecutorService
        worker.submit {}.get(5,java.util.concurrent.TimeUnit.SECONDS)
        val engine=MainActivity::class.java.getDeclaredField("engine").apply { isAccessible=true }.get(activity) as ChatEngine
        val persisted=StateJson.decode(StateJson.encode(engine.state))
        assertEquals(4,persisted.drafts["all"]?.cursor)
        val (reopened,newRoot)=fixture("home")
        field(reopened,"shown",persisted); field(reopened,"engine",ChatEngine(persisted)); field(reopened,"renderedRoute","")
        MainActivity::class.java.getDeclaredMethod("restoreSession",ChatState::class.java).apply { isAccessible=true }.invoke(reopened,persisted)
        call(reopened,"render"); layout(newRoot)
        assertTrue(descendants(newRoot).any { it.contentDescription?.startsWith("Guide step 5:")==true })
        MainActivity::class.java.getDeclaredMethod("moveGuide",Int::class.javaPrimitiveType).apply { isAccessible=true }.invoke(reopened,-1)
        layout(newRoot); looper.idleFor(java.time.Duration.ofMillis(250)); layout(newRoot)
        assertEquals("An unfinished note",descendants(newRoot).filterIsInstance<android.widget.EditText>().single().text.toString())
        assertEquals(4,descendants(newRoot).filterIsInstance<android.widget.EditText>().single().selectionEnd)
        descendants(newRoot).first { it.contentDescription=="Pause guide" }.performClick(); layout(newRoot)
        assertFalse(descendants(newRoot).any { it is GuideCoach })
        assertEquals("settings",MainActivity::class.java.getDeclaredField("route").apply { isAccessible=true }.get(reopened))
        MainActivity::class.java.getDeclaredMethod("startGuide",Boolean::class.javaPrimitiveType,Int::class.javaObjectType).apply { isAccessible=true }.invoke(reopened,false,null)
        layout(newRoot)
        assertTrue(descendants(newRoot).any { it.contentDescription?.startsWith("Guide step 4:")==true })
        assertEquals(4,descendants(newRoot).filterIsInstance<android.widget.EditText>().single().selectionEnd)
    }
    @Test fun leftForwardArrowUsesTheDisplayedMessageAndKeepsItsOriginal() {
        val (activity,root)=fixture()
        var forwarded=emptyList<Message>()
        field(activity,"forwardAction",{ notes:List<Message> -> forwarded=notes })
        val wrapper=descendants(root).filterIsInstance<ReplySwipe>().first()
        val arrow=descendants(wrapper).first { it.contentDescription=="Forward message" }
        val bubble=descendants(wrapper).filterIsInstance<BubbleLayout>().single()
        assertTrue(arrow.right<=bubble.left)
        arrow.performClick()
        assertEquals(listOf(messages.first()),forwarded)
        val engine=MainActivity::class.java.getDeclaredField("engine").apply { isAccessible=true }.get(activity) as ChatEngine
        assertEquals(messages,engine.state.messages)
    }
    private fun layout(root:View) { val density=root.resources.displayMetrics.density; val width=(393*density).toInt(); val height=(830*density).toInt(); root.measure(View.MeasureSpec.makeMeasureSpec(width,View.MeasureSpec.EXACTLY),View.MeasureSpec.makeMeasureSpec(height,View.MeasureSpec.EXACTLY)); root.layout(0,0,width,height) }
    private fun descendants(view:View):List<View> = listOf(view)+(if(view is ViewGroup)(0 until view.childCount).flatMap { descendants(view.getChildAt(it)) } else emptyList())
    private fun field(activity:MainActivity,name:String,value:Any) { MainActivity::class.java.getDeclaredField(name).apply { isAccessible=true }.set(activity,value) }
    private fun call(activity:MainActivity,name:String) { MainActivity::class.java.getDeclaredMethod(name).apply { isAccessible=true }.invoke(activity) }
    @Suppress("UNCHECKED_CAST") private fun selected(activity:MainActivity)=MainActivity::class.java.getDeclaredField("selected").apply { isAccessible=true }.get(activity) as Set<String>
}
