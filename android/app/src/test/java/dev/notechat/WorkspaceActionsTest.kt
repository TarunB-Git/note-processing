package dev.notechat

import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

class WorkspaceActionsTest {
    private val categories=listOf(Category("todos","To-do","Actions"),Category("ideas","Ideas","Possibilities"))
    @Test fun splitPreservesPositionTimeAndRepliesButRevokesTraining() {
        val original=Message(id="original",text="Buy milk\nBuild a shelf",created=100,approved=true,category="todos",starred=true)
        val engine=ChatEngine(ChatState(categories,listOf(Message(id="before"),original,Message(id="reply",replyTo="original"))))
        val parts=engine.split(original.id,listOf("Buy milk","Build a shelf"))
        assertEquals(listOf("before","original",parts[1].id,"reply"),engine.state.messages.map { it.id })
        assertTrue(parts.all { it.created==100L && !it.approved && it.category==null && it.starred })
        assertEquals("original",engine.state.messages.last().replyTo)
        assertTrue(engine.confirmed().isEmpty())
        val rest=engine.split(parts[1].id,listOf("Build","a shelf"))
        assertEquals(rest.map { it.id },engine.state.messages.subList(2,4).map { it.id })
    }
    @Test fun mergeUsesInboxOrderRetargetsRepliesAndRejectsAttachmentLoss() {
        val first=Message(id="first",text="one",created=100,approved=true,category="ideas")
        val second=Message(id="second",text="two",created=200,starred=true)
        val engine=ChatEngine(ChatState(categories,listOf(first,Message(id="between"),second,Message(id="reply",replyTo="second"))))
        val merged=engine.merge(linkedSetOf("second","first"))
        assertEquals("one\n\ntwo",merged.text); assertEquals("first",merged.id); assertEquals(100L,merged.created)
        assertFalse(merged.approved); assertTrue(merged.starred)
        assertEquals(listOf("first","between","reply"),engine.state.messages.map { it.id })
        assertEquals("first",engine.state.messages.last().replyTo)
        engine.state=engine.state.copy(messages=engine.state.messages+Message(id="media",attachment=Attachment("file","photo.jpg","image/jpeg",12)))
        assertThrows(IllegalArgumentException::class.java) { engine.merge(setOf("first","media")) }
        assertNotNull(engine.state.messages.first { it.id=="media" }.attachment)
    }
    @Test fun completionAndStarArchivesHonorExactDayBoundaryAndRestore() {
        val engine=ChatEngine(ChatState(categories,listOf(Message(id="task",text="Buy milk",category="todos",starred=true))))
        engine.complete("task",true,100); engine.archiveStars(setOf("task"),100)
        val message=engine.state.messages.single()
        assertTrue(message.visible(100+ARCHIVE_DELAY_MS-1)); assertFalse(message.starArchived(100+ARCHIVE_DELAY_MS-1))
        assertFalse(message.visible(100+ARCHIVE_DELAY_MS)); assertTrue(message.starArchived(100+ARCHIVE_DELAY_MS))
        assertFalse(message.copy(done=false,completedAt=null).visible(100+ARCHIVE_DELAY_MS))
        engine.complete("task",false); engine.update("task") { it.copy(starArchiveAt=null) }
        assertTrue(engine.state.messages.single().visible()); assertFalse(engine.state.messages.single().starArchived())
    }
    @Test fun movingCompletedTaskToAnotherCategoryCancelsTaskArchival() {
        val engine=ChatEngine(ChatState(categories,listOf(Message(id="task",text="A new approach",category="todos",done=true,completedAt=1))))
        engine.tag("task","ideas")
        assertTrue(engine.state.messages.single().visible(ARCHIVE_DELAY_MS+1))
        assertFalse(engine.state.messages.single().done)
    }
    @Test fun deleteRangeIsInclusiveAndUsesStoredDatesAcrossCategories() {
        val engine=ChatEngine(ChatState(categories,listOf(Message(id="outside",created=99),Message(id="start",created=100),Message(id="end",created=200,category="todos"),Message(id="after",created=201))))
        engine.deleteRange(100,200,300)
        assertEquals(setOf("outside","after"),engine.active().map { it.id }.toSet())
        assertEquals(300L,engine.state.messages.first { it.id=="end" }.deleted)
        engine.restore("end",301); assertEquals(3,engine.active().size)
    }
    @Test fun archivesTranscriptsAndImportPreferencesSurviveEncryptedStateCodec() {
        val state=ChatState(categories,listOf(Message(text="Call the dentist",done=true,completedAt=100,starred=true,starArchiveAt=200,transcribed=true,
            attachment=Attachment("file","Voice.m4a","audio/mp4",10))),preferences=Preferences(true,"dark"),navigation=Navigation("todos","task",12,-30,0))
        assertEquals(state,StateJson.decode(StateJson.encode(state)))
        val old=org.json.JSONObject(StateJson.encode(state).toString(Charsets.UTF_8)).apply {
            remove("preferences"); remove("navigation"); getJSONArray("messages").getJSONObject(0).apply { remove("completedAt"); remove("starArchiveAt"); remove("transcribed") }
        }
        val decoded=StateJson.decode(old.toString().toByteArray())
        assertFalse(decoded.preferences.preserveImportDates); assertEquals("system",decoded.preferences.appearance)
        assertEquals(Navigation(),decoded.navigation); assertNull(decoded.messages.single().completedAt); assertFalse(decoded.messages.single().transcribed)
    }
    @Test fun reviewedVoiceTextTrainsOnlyAfterApprovalAndImagesNeverTrain() {
        val voice=Message(id="voice",text="Call the dentist",category="todos",transcribed=true,attachment=Attachment("file","Voice.m4a","audio/mp4",10))
        val photo=Message(id="photo",text="Call the dentist",category="todos",transcribed=true,attachment=Attachment("photo","photo.jpg","image/jpeg",10))
        val engine=ChatEngine(ChatState(categories,listOf(voice,photo)))
        assertTrue(engine.confirmed().isEmpty()); engine.approve("voice"); engine.approve("photo")
        assertEquals(listOf("voice"),engine.confirmed().map { it.id })
        engine.edit("voice","Call my doctor"); assertTrue(engine.confirmed().isEmpty())
    }
    @Test fun reclassificationRefreshesManualPendingTextWithoutLearningOrTouchingApprovals() {
        val approved=Message(id="approved",text="Book the trip",category="todos",approved=true)
        val photo=Message(id="photo",category="ideas",attachment=Attachment("photo","Photo.jpg","image/jpeg",10))
        val deleted=Message(id="deleted",text="Discarded note",deleted=100)
        val archived=Message(id="archived",text="Finished task",category="todos",done=true,completedAt=1)
        val manual=Message(id="manual",text="Build a shelf",category="ideas",manual=true,rejected=setOf("todos"))
        val voice=Message(id="voice",text="Call the shop",transcribed=true,attachment=Attachment("voice","Voice.m4a","audio/mp4",10))
        val engine=ChatEngine(ChatState(categories,listOf(approved,photo,deleted,archived,manual,voice)))
        val seen=mutableListOf<String>()
        val count=engine.reclassifyPending(ARCHIVE_DELAY_MS+2) {
            seen.add(it.id); assertFalse(it.manual); assertTrue(it.rejected.isEmpty()); Suggestion("todos","Current learner")
        }
        assertEquals(2,count); assertEquals(listOf("manual","voice"),seen)
        assertEquals(listOf(approved,photo,deleted,archived),engine.state.messages.take(4))
        assertTrue(engine.state.messages.takeLast(2).all { it.category=="todos" && !it.approved })
        assertEquals(listOf(approved),engine.confirmed())
    }
    @Test fun speechDecoderMixesStereoAndDoesNotDiscardFrames() {
        val input=ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).apply { putShort(1000); putShort(3000); putShort(-2000); putShort(2000); flip() }
        assertArrayEquals(floatArrayOf(2000f,0f),VoiceTranscriber.monoPcm(input,2,false),0f)
        val floating=ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).apply { putFloat(.5f); putFloat(-.5f); flip() }
        assertArrayEquals(floatArrayOf(0f),VoiceTranscriber.monoPcm(floating,2,true),0f)
    }
    @Test fun speechModelExtractionRejectsPathsOutsideStagingDirectory() {
        val directory=java.nio.file.Files.createTempDirectory("speech-extract").toFile()
        try {
            val archive=File(directory,"bad.zip")
            ZipOutputStream(archive.outputStream()).use { it.putNextEntry(ZipEntry("../escape")); it.write(byteArrayOf(1)); it.closeEntry() }
            val stage=File(directory,"stage").apply { mkdirs() }
            assertThrows(IllegalArgumentException::class.java) { VoiceTranscriber.unpack(archive,stage) }
            assertFalse(File(directory,"escape").exists())
        } finally { directory.deleteRecursively() }
    }
}
