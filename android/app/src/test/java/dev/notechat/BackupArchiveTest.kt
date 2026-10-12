package dev.notechat

import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.nio.file.Files
import java.util.UUID
import java.util.zip.ZipEntry
import java.util.zip.ZipFile
import java.util.zip.ZipOutputStream

@RunWith(RobolectricTestRunner::class)
@Config(sdk=[35])
class BackupArchiveTest {
    private val categories=listOf(Category("thoughts","Thoughts","Observations"))
    @Test fun encryptedBackupRestoresMediaAndApprovalsWithoutDeviceFolderAccess() {
        val dir=Files.createTempDirectory("backup").toFile()
        try {
            val bytes="private attachment".toByteArray()
            val attachment=Attachment(UUID.randomUUID().toString(),"note.txt","text/plain",bytes.size.toLong())
            val state=ChatState(categories,listOf(Message(text="private note",category="thoughts",approved=true,attachment=attachment)),
                markdown=MarkdownState("content://old-device"),draft="private draft",chats=listOf(Chat("journal","Journal")),
                drafts=mapOf("all" to ComposerDraft("unfinished private draft",cursor=4)),
                locations=mapOf("all" to Navigation("all")),guide=GuideProgress(true,4),homeTab="chats")
            val file=java.io.File(dir,"backup.zip")
            BackupArchive.write(file,state,"correct-password".toCharArray()) { bytes.inputStream() }
            ZipFile(file).use { zip -> assertFalse(zip.getInputStream(zip.getEntry("workspace.enc")).readBytes().toString(Charsets.UTF_8).contains("private note")) }
            assertThrows(Exception::class.java) { BackupArchive.read(file,"wrong-password".toCharArray(),dir) }
            val restored=BackupArchive.read(file,"correct-password".toCharArray(),dir)
            assertEquals(state.messages,restored.state.messages)
            assertEquals(state.chats,restored.state.chats)
            assertNull(restored.state.markdown.tree); assertEquals("",restored.state.draft)
            assertTrue(restored.state.drafts.isEmpty()); assertTrue(restored.state.locations.isEmpty())
            assertFalse(restored.state.guide.active); assertEquals("home",restored.state.homeTab)
            assertArrayEquals(bytes,restored.media.getValue(attachment.file).readBytes())
            val merged=BackupArchive.merge(state,restored)
            assertEquals(1,merged.messages.size)
            assertTrue(restored.fingerprint in merged.importedArchives)
            restored.close(); assertFalse(restored.staging.exists())
        } finally { dir.deleteRecursively() }
    }
    @Test fun changingCiphertextRejectsTheWholeRestore() {
        val dir=Files.createTempDirectory("backup-tamper").toFile()
        try {
            val original=java.io.File(dir,"original.zip"); val broken=java.io.File(dir,"broken.zip")
            BackupArchive.write(original,ChatState(categories,listOf(Message(text="keep this private"))),"correct-password".toCharArray()) { error("No media") }
            ZipFile(original).use { zip -> ZipOutputStream(broken.outputStream()).use { output ->
                zip.entries().asSequence().forEach { entry ->
                    val bytes=zip.getInputStream(entry).readBytes()
                    if(entry.name=="workspace.enc")bytes[bytes.lastIndex]=(bytes.last().toInt() xor 1).toByte()
                    output.putNextEntry(ZipEntry(entry.name)); output.write(bytes); output.closeEntry()
                }
            } }
            assertThrows(Exception::class.java) { BackupArchive.read(broken,"correct-password".toCharArray(),dir) }
            assertFalse(dir.listFiles()!!.any { it.name.startsWith("restore-") })
        } finally { dir.deleteRecursively() }
    }
    @Test fun whatsappZipChoosesTranscriptWithoutExtractingPaths() {
        val dir=Files.createTempDirectory("transcript").toFile()
        try {
            val file=java.io.File(dir,"chat.zip")
            ZipOutputStream(file.outputStream()).use { zip ->
                zip.putNextEntry(ZipEntry("../outside.txt")); zip.write("8/10/26, 10:00 - Me: Download books\n8/10/26, 10:01 - Me: Another note".toByteArray()); zip.closeEntry()
                zip.putNextEntry(ZipEntry("picture.jpg")); zip.write(byteArrayOf(1,2)); zip.closeEntry()
            }
            val candidates=ZipTranscript.read(file)
            assertEquals(1,candidates.size); assertEquals(2,candidates.single().messages)
            assertFalse(java.io.File(dir.parentFile,"outside.txt").exists())
        } finally { dir.deleteRecursively() }
    }
    @Test fun modelArchivesRejectNotesAndRoundTripOnlyCompatibleWeights() {
        val dir=Files.createTempDirectory("model-backup").toFile()
        try {
            val approved=(0..5).map { Message(text="Observation $it",category="thoughts",approved=true) }
            val classifier=TextClassifier(null).apply { configure(categories,approved); train() }
            val model=classifier.learned()!!
            val file=java.io.File(dir,"model.zip")
            assertThrows(IllegalArgumentException::class.java) {
                BackupArchive.write(file,ChatState(categories,approved,model=model),"correct-password".toCharArray(),"model") { error("No media") }
            }
            BackupArchive.write(file,ChatState(categories,model=model),"correct-password".toCharArray(),"model") { error("No media") }
            val restored=BackupArchive.read(file,"correct-password".toCharArray(),dir)
            try {
                assertTrue(restored.state.messages.isEmpty()); assertTrue(restored.state.splits.isEmpty()); assertNull(restored.state.quiz)
                assertTrue(TextClassifier.compatible(restored.state.model!!,categories,null))
                assertFalse(TextClassifier.compatible(restored.state.model!!,categories,"another encoder"))
                assertFalse(TextClassifier.compatible(restored.state.model!!,categories.map { it.copy(description="Another definition") },null))
            } finally { restored.close() }
        } finally { dir.deleteRecursively() }
    }

}
