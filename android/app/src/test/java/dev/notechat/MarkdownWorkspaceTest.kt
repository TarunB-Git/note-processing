package dev.notechat

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class MarkdownWorkspaceTest {
    private val categories=listOf(Category("thoughts","Thoughts","Private description",listOf("Private example")),Category("todos","To-do","Actions"))
    private fun workspace(): ChatState {
        val state=ChatState(categories,listOf(Message(id="one",text="A reflection.\n\n## A heading\n\n```md\n[[Example]]\n```",category="thoughts",approved=true),Message(id="two",text="Buy books",category="todos",approved=true)))
        val files=mapOf("Thoughts.md" to MarkdownWorkspace.render(state,"thoughts"),"To-do.md" to MarkdownWorkspace.render(state,"todos"))
        return state.copy(markdown=MarkdownState("content://fixture/tree/root",files))
    }
    @Test fun roundTripKeepsMultilineMarkdownAndDoesNotCreateApprovals() {
        val state=workspace()
        val doc=MarkdownWorkspace.parse(state.markdown.baseline.getValue("Thoughts.md"),state)
        assertEquals(state.messages.first().text,doc.entries.single().text)
        assertEquals(state,MarkdownWorkspace.merge(state,state.markdown.baseline).state)
        assertEquals(state,StateJson.decode(StateJson.encode(state)))
    }
    @Test fun readableFilenamesAndAliasesKeepLinksValidAcrossCategoryRenames() {
        val state=workspace()
        assertEquals("Thoughts.md",MarkdownWorkspace.filename(state,"thoughts"))
        val renamed=state.copy(categories=state.categories.map { if(it.id=="thoughts")it.copy(title="Reflections") else it })
        assertEquals("Thoughts.md",MarkdownWorkspace.filename(renamed,"thoughts"))
        assertTrue(MarkdownWorkspace.render(renamed,"thoughts").contains("aliases: [\"Reflections\"]"))
        val collision=ChatState(listOf(Category("a","Ideas!","First"),Category("b","Ideas?","Second")))
        assertNotEquals(MarkdownWorkspace.filename(collision,"a"),MarkdownWorkspace.filename(collision,"b"))
    }
    @Test fun editsRevokeApprovalAndCheckboxChangesPreserveTextAndApproval() {
        val state=workspace()
        val files=state.markdown.baseline.toMutableMap()
        files["Thoughts.md"]=files.getValue("Thoughts.md").replace("A reflection.","A revised reflection.")
        files["To-do.md"]=files.getValue("To-do.md").replace("- [ ] Buy","- [x] Buy")
        val result=MarkdownWorkspace.merge(state,files,1000).state
        assertFalse(result.messages[0].approved)
        assertTrue(result.messages[0].edited)
        assertTrue(result.messages[1].approved)
        assertTrue(result.messages[1].done)
        assertEquals(1000L,result.messages[1].completedAt)
    }
    @Test fun appAndEditorConflictsRetainBothVersionsWithoutOverwritingAppText() {
        val state=workspace()
        val changed=state.copy(messages=state.messages.map { if(it.id=="one")it.copy(text="Edited in app",approved=false) else it })
        val files=state.markdown.baseline.toMutableMap()
        files["Thoughts.md"]=files.getValue("Thoughts.md").replace("A reflection.","Edited in editor")
        val merged=MarkdownWorkspace.merge(changed,files)
        assertEquals(setOf("Thoughts.md"),merged.conflicts)
        assertEquals("Edited in app",merged.state.messages.first().text)
    }
    @Test fun movingABlockChangesCategoryWithoutDuplicatingTheMessage() {
        val state=workspace()
        val thoughts=state.markdown.baseline.getValue("Thoughts.md")
        val block=thoughts.substring(thoughts.indexOf("<!-- notechat:message"))
        val files=state.markdown.baseline.toMutableMap()
        files["Thoughts.md"]=thoughts.substringBefore("<!-- notechat:message")
        files["To-do.md"]=files.getValue("To-do.md")+block
        val result=MarkdownWorkspace.merge(state,files).state
        assertEquals(2,result.messages.size)
        assertEquals("todos",result.messages.first().category)
        assertTrue(result.messages.first().approved)
        assertNull(result.messages.first().deleted)
    }
    @Test fun removedBlocksGoToTrashButMissingWholeFilesDoNotDeleteMessages() {
        val state=workspace()
        val removed=state.markdown.baseline.toMutableMap()
        removed["Thoughts.md"]=removed.getValue("Thoughts.md").substringBefore("<!-- notechat:message")
        assertEquals(1000L,MarkdownWorkspace.merge(state,removed,1000).state.messages.first().deleted)
        assertEquals(state,MarkdownWorkspace.merge(state,emptyMap()).state)
    }
    @Test fun appendedParagraphsAndFencedCodeKeepTheChosenCategory() {
        val state=workspace()
        val files=state.markdown.baseline.toMutableMap()
        files["Thoughts.md"]=files.getValue("Thoughts.md")+"New thought\n\n```python\nvalue = 1\n\nprint(value)\n```\n"
        val result=MarkdownWorkspace.merge(state,files,2000).state
        assertEquals(4,result.messages.size)
        assertTrue(result.messages.takeLast(2).all { it.approved })
        assertTrue(result.messages.last().text.contains("\n\nprint"))
        assertTrue(result.messages.takeLast(2).all { it.created==2000L && it.category=="thoughts" })
    }
    @Test fun damagedMarkersAndDuplicateIdsStopSyncBeforeChangingState() {
        val state=workspace()
        val original=state.markdown.baseline.getValue("Thoughts.md")
        for(source in listOf(original.replace("<!-- /notechat -->",""),original+original.substringAfter("# Thoughts\n"))) {
            assertThrows(IllegalArgumentException::class.java) { MarkdownWorkspace.merge(state,mapOf("Thoughts.md" to source)) }
        }
        assertTrue(state.messages.first().approved)
    }
    @Test fun mediaBlocksKeepAttachmentIdentityAndAllowCaptionEditing() {
        val attachment=Attachment("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee","Photo.png","image/png",30)
        val state=ChatState(categories,listOf(Message(id="photo",text="Caption",category="thoughts",attachment=attachment)))
        val source=MarkdownWorkspace.render(state,"thoughts")
        val base=state.copy(markdown=MarkdownState(baseline=mapOf("Thoughts.md" to source)))
        val result=MarkdownWorkspace.merge(base,mapOf("Thoughts.md" to source.replace("Caption","Updated caption")))
        assertEquals(attachment,result.state.messages.single().attachment)
        assertEquals("Updated caption",result.state.messages.single().text)
    }
    @Test fun shareableReportExcludesPrivateTextMetadataAndLearnedWeights() {
        val state=workspace().copy(model=PersonalHead("Private signature",listOf(listOf(1.234567f)),listOf(0f)),splits=mapOf("Private split text" to "train"))
        val report=ModelReport.encode(state,null)
        listOf("Private description","Private example","Private signature","Private split text","A reflection","Buy books","1.234567","Thoughts").forEach { assertFalse(report.contains(it)) }
        assertEquals(2,JSONObject(report).getInt("approvedCount"))
        assertEquals("starter",JSONObject(report).getString("activeHead"))
        assertTrue(JSONObject(report).getBoolean("storedPersonalHead"))
    }
    @Test fun widgetProjectionContainsOnlyVisibleTasksAndNoLearningMetadata() {
        val now=2*ARCHIVE_DELAY_MS
        val state=ChatState(categories,listOf(Message(id="active",text="Do this",category="todos",reason="Private reason",approved=true),
            Message(id="done",category="todos",done=true,completedAt=0),Message(id="deleted",category="todos",deleted=1),Message(id="thought",category="thoughts")))
        val projected=WidgetTasks.project(state,now)
        assertEquals(listOf("active"),projected.map { it.id })
        assertFalse(projected.single().approved)
        assertEquals("",projected.single().reason)
        val scheduled=state.copy(messages=listOf(Message(id="scheduled",text="Task",category="todos",starred=true,starArchiveAt=now)))
        assertFalse(WidgetTasks.project(scheduled,now).single().visible(now+ARCHIVE_DELAY_MS))
    }
    @Test fun pinsExpireLimitToThreeAndSurviveStateSerialization() {
        val engine=ChatEngine(ChatState(categories,(1..4).map { Message(id="$it",text="Note $it") }))
        (1..3).forEach { engine.pin("$it",1000,10) }
        assertThrows(IllegalArgumentException::class.java) { engine.pin("4",1000,10) }
        engine.pin("1",null,10); engine.pin("4",1000,10)
        assertEquals(engine.state,StateJson.decode(StateJson.encode(engine.state)))
        assertEquals(0,engine.state.messages.count { (it.pinnedUntil ?: 0)>1001 })
    }
}
