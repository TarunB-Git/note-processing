package dev.notechat

import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk=[35])
class PrivateChatTest {
    private val categories=listOf(Category("thoughts","Thoughts","Observations"))
    @Test fun oldWorkspacesKeepPrivateMessagesSeparateAndOptInSurvivesRestore() {
        val private=Message(text="Journal entry",chatId="journal")
        val note=Message(text="Inbox note")
        val state=ChatState(categories,listOf(private,note))
        assertFalse(private.inInbox(state.preferences)); assertTrue(note.inInbox(state.preferences))
        val enabled=StateJson.decode(StateJson.encode(state.copy(preferences=Preferences(includePrivateChatsInInbox=true))))
        assertTrue(private.inInbox(enabled.preferences))
        val legacy=org.json.JSONObject(StateJson.encode(enabled).toString(Charsets.UTF_8))
        legacy.getJSONObject("preferences").remove("includePrivateChatsInInbox")
        assertFalse(StateJson.decode(legacy.toString().toByteArray()).preferences.includePrivateChatsInInbox)
    }
    @Test fun privateChatsAreExcludedUntilManuallyCategorized() {
        val note=Message(text="Private note",chatId="journal")
        val engine=ChatEngine(ChatState(categories,listOf(note),chats=listOf(Chat("journal","Journal"))))
        assertFalse(note.trainable()); assertTrue(engine.confirmed().isEmpty())
        engine.tag(note.id,"thoughts")
        assertNull(engine.state.messages.single().chatId)
        assertTrue(engine.state.messages.single().approved)
        assertEquals(1,engine.confirmed().size)
    }
    @Test fun promotingAChatPreservesRecordsAndApprovesTheAssignedCategory() {
        val note=Message(id="one",text="A garden idea",chatId="garden")
        val engine=ChatEngine(ChatState(categories,listOf(note),chats=listOf(Chat("garden","Garden"))))
        engine.promoteChat("garden","Garden plans")
        assertTrue(engine.state.chats.isEmpty())
        assertEquals("one",engine.state.messages.single().id)
        assertEquals("garden",engine.state.messages.single().category)
        assertTrue(engine.state.messages.single().approved)
        assertEquals(2,engine.state.categories.size)
    }
    @Test fun forwardingCreatesANewRecordAndKeepsTheOriginal() {
        val note=Message(text="Observation",category="thoughts",approved=true,done=false,pinnedUntil=Long.MAX_VALUE)
        val engine=ChatEngine(ChatState(categories,listOf(note),chats=listOf(Chat("journal","Journal"))))
        val copy=engine.forward(setOf(note.id),"chat:journal").single()
        assertEquals(note,engine.state.messages.first())
        assertNotEquals(note.id,copy.id); assertNull(copy.category); assertFalse(copy.approved); assertFalse(copy.trainable()); assertNull(copy.pinnedUntil)
        val categorized=engine.forward(setOf(copy.id),"thoughts").single()
        assertTrue(categorized.approved); assertNull(categorized.chatId)
    }
    @Test fun classificationOffDoesNotInvokePredictor() {
        val engine=ChatEngine(ChatState(categories,listOf(Message(text="Pending")),preferences=Preferences(classification=false)))
        assertEquals(0,engine.reclassifyPending { error("Classifier must stay off") })
    }
    @Test fun quizReportsOnlyFrozenPredictionsAgainstAnsweredLabels() {
        val quiz=Quiz(listOf(QuizItem("A","thoughts","thoughts"),QuizItem("B",null,"thoughts"),QuizItem("C","thoughts")))
        assertEquals(.5,Diagnostics.accuracy(quiz)!!,.00001)
        val restored=StateJson.decode(StateJson.encode(ChatState(categories,quiz=quiz,editDraftId="one",preEditDraft="Unsent")))
        assertEquals(quiz,restored.quiz); assertEquals("one",restored.editDraftId); assertEquals("Unsent",restored.preEditDraft)
    }
    @Test fun remindersUseFutureLocalDatesAndRejectInvalidDates() {
        val calendar=java.util.Calendar.getInstance().apply { set(2026,9,12,14,0,0); set(java.util.Calendar.MILLISECOND,0) }
        val now=calendar.timeInMillis
        val time=ReminderTime.detect("Call tomorrow at 5pm",now)!!
        calendar.timeInMillis=time
        assertEquals(13,calendar.get(java.util.Calendar.DAY_OF_MONTH)); assertEquals(17,calendar.get(java.util.Calendar.HOUR_OF_DAY))
        assertNull(ReminderTime.detect("Call today at 1pm",now)); assertNull(ReminderTime.detect("Call 2026-02-30 at 17:00",now))
    }
    @Test fun removingACategoryConfirmsAnExplicitReplacementAndClearsAnOldQuiz() {
        val classes=categories+Category("ideas","Ideas","Possibilities")
        val note=Message(text="Observation",category="ideas",approved=true)
        val engine=ChatEngine(ChatState(classes,listOf(note),quiz=Quiz(listOf(QuizItem("Sample","ideas","ideas")))))
        engine.removeCategory("ideas","thoughts")
        assertTrue(engine.state.messages.single().approved)
        assertEquals("thoughts",engine.state.messages.single().category)
        assertNull(engine.state.quiz)
    }

}
