package dev.notechat

import org.junit.Assert.*
import org.junit.Test

class ChatPolicyTest {
    private val categories=listOf(Category("thoughts","Thoughts","Observations"),Category("todos","To-do","Actions"),Category("work","Work","Company information"))
    @Test fun rejectionDoesNotTeachAndNeverRepeatsRejectedCategories() {
        val note=Message(text="Prepare the slides",category="thoughts")
        val engine=ChatEngine(ChatState(categories,listOf(note)))
        engine.reject(note.id) { pending -> assertEquals(setOf("thoughts"),pending.rejected); Suggestion("todos","Action") }
        assertTrue(engine.confirmed().isEmpty())
        engine.reject(note.id) { pending -> assertEquals(setOf("thoughts","todos"),pending.rejected); Suggestion("work","Work") }
        engine.approve(note.id)
        assertEquals("work",engine.confirmed().single().category)
    }
    @Test fun manualTagsStayUnconfirmedUntilThumbsUp() {
        val note=Message(text="Roadmap",category="thoughts")
        val engine=ChatEngine(ChatState(categories,listOf(note)))
        engine.tag(note.id,"work")
        assertTrue(engine.confirmed().isEmpty())
        engine.approve(note.id)
        assertEquals("work",engine.confirmed().single().category)
        engine.edit(note.id,"The roadmap was revised")
        assertTrue(engine.confirmed().isEmpty())
        assertNull(engine.state.messages.single().category)
    }
    @Test fun mediaNeverEntersTextTraining() {
        val note=Message(text="An image caption",category="work",attachment=Attachment("id","photo.jpg","image/jpeg",20))
        val engine=ChatEngine(ChatState(categories,listOf(note)))
        engine.approve(note.id)
        assertTrue(engine.confirmed().isEmpty())
    }
    @Test fun deletingRemovesAllViewsAndLearningButRestoreKeepsTheOriginalMessage() {
        val note=Message(text="Send the agenda",category="todos",approved=true,starred=true)
        val engine=ChatEngine(ChatState(categories,listOf(note)))
        engine.delete(setOf(note.id),1000)
        assertTrue(engine.active().isEmpty()); assertTrue(engine.confirmed().isEmpty())
        assertEquals(note.id,engine.state.messages.single().id)
        engine.restore(note.id,1000+RETENTION_MS-1)
        assertEquals(note,engine.active().single())
    }
    @Test fun ninetyDayBoundaryPreventsRestoreAndExpiresOnlyDeletedMessages() {
        val deleted=Message(text="Deleted",deleted=1000)
        val active=Message(text="Keep this",created=0)
        val engine=ChatEngine(ChatState(categories,listOf(deleted,active)))
        try { engine.restore(deleted.id,1000+RETENTION_MS); fail("Expired messages must not restore") } catch(_:IllegalArgumentException) { }
        assertEquals(listOf(deleted),engine.expire(1000+RETENTION_MS))
        assertEquals(listOf(active),engine.active())
    }
    @Test fun removingCategoryKeepsActiveAndDeletedMessagesWithoutOldApproval() {
        val engine=ChatEngine(ChatState(categories,listOf(Message(text="Update",category="work",approved=true),Message(text="Older",category="work",deleted=1000))))
        engine.removeCategory("work","thoughts")
        assertEquals(2,engine.state.messages.size)
        assertTrue(engine.state.messages.all { it.category=="thoughts" && !it.approved })
        assertTrue(engine.state.categories.none { it.id=="work" })
        assertTrue(engine.confirmed().isEmpty())
    }
    @Test fun revokingApprovalInvalidatesFittedHead() {
        val note=Message(text="Roadmap",category="work",approved=true)
        val engine=ChatEngine(ChatState(categories,listOf(note),model=PersonalHead("test",emptyList(),emptyList())))
        engine.tag(note.id,"thoughts")
        assertNull(engine.state.model)
    }
    @Test fun latestApprovalWinsForRepeatedText() {
        val engine=ChatEngine(ChatState(categories,listOf(Message(text="Roadmap",category="thoughts",approved=true),Message(text=" roadmap ",category="work",approved=true))))
        assertEquals("work",engine.confirmed().single().category)
    }
    @Test fun exportedChatOmitsDeletedMessagesAndKeepsTaskCompletionAndMediaPaths() {
        val a=Attachment("abc","../holiday photo.jpg","image/jpeg",10)
        val text=ChatExport.render("To-do",listOf(Message(text="Book travel",category="todos",done=true),Message(text="Private deleted text",deleted=1000),Message(attachment=a)),categories,true)
        assertTrue(text.contains("- [x] Book travel")); assertFalse(text.contains("Private deleted text"))
        assertTrue(text.contains(ChatExport.mediaName(a))); assertFalse(ChatExport.mediaName(a).contains("/../"))
    }
    @Test fun snapshotCodecPreservesApprovalRejectionsAttachmentsAndDeletion() {
        val state=ChatState(categories,listOf(Message(text="Hello 😀",category="work",approved=true,rejected=setOf("thoughts"),attachment=Attachment("id","clip.m4a","audio/mp4",20,900,listOf(5,18,4)),deleted=1000,sender="Alex")),splits=mapOf("hello" to "test"),draft="Unsent note",replyDraft="reply-id")
        assertEquals(state,StateJson.decode(StateJson.encode(state)))
    }
}
