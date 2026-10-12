package dev.notechat

import android.content.Context
import android.content.pm.ProviderInfo
import android.database.Cursor
import android.database.MatrixCursor
import android.net.Uri
import android.os.CancellationSignal
import android.os.ParcelFileDescriptor
import android.provider.DocumentsContract
import android.provider.DocumentsProvider
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config
import org.robolectric.shadows.ShadowContentResolver
import java.io.File
import java.io.FileNotFoundException

@RunWith(RobolectricTestRunner::class)
@Config(sdk=[35])
class MarkdownFolderTest {
    class Files : DocumentsProvider() {
        lateinit var root: File
        var failWrite=false
        var writes=0
        override fun onCreate()=true
        override fun queryRoots(projection: Array<out String>?): Cursor = MatrixCursor(arrayOf(DocumentsContract.Root.COLUMN_ROOT_ID)).apply { addRow(arrayOf("root")) }
        private fun file(id: String)=if(id=="root")root else File(root,id.removePrefix("root/"))
        override fun queryDocument(documentId: String, projection: Array<out String>?): Cursor {
            val columns=projection ?: arrayOf(DocumentsContract.Document.COLUMN_DOCUMENT_ID,DocumentsContract.Document.COLUMN_DISPLAY_NAME,DocumentsContract.Document.COLUMN_MIME_TYPE)
            return MatrixCursor(columns).apply { addRow(columns.map { column -> when(column) {
                DocumentsContract.Document.COLUMN_DOCUMENT_ID->documentId
                DocumentsContract.Document.COLUMN_DISPLAY_NAME->file(documentId).name
                DocumentsContract.Document.COLUMN_SIZE->file(documentId).length()
                DocumentsContract.Document.COLUMN_MIME_TYPE->if(file(documentId).isDirectory)DocumentsContract.Document.MIME_TYPE_DIR else "text/markdown"
                else->0
            } }.toTypedArray()) }
        }
        override fun queryChildDocuments(parentDocumentId: String, projection: Array<out String>?, sortOrder: String?): Cursor {
            val columns=projection ?: arrayOf(DocumentsContract.Document.COLUMN_DOCUMENT_ID,DocumentsContract.Document.COLUMN_DISPLAY_NAME)
            return MatrixCursor(columns).apply { file(parentDocumentId).listFiles()?.forEach { child -> addRow(columns.map { column -> when(column) {
                DocumentsContract.Document.COLUMN_DOCUMENT_ID->"root/"+child.relativeTo(root).invariantSeparatorsPath
                DocumentsContract.Document.COLUMN_DISPLAY_NAME->child.name
                else->0
            } }.toTypedArray()) } }
        }
        override fun createDocument(parentDocumentId: String, mimeType: String, displayName: String): String {
            val child=File(file(parentDocumentId),displayName)
            if(mimeType==DocumentsContract.Document.MIME_TYPE_DIR)child.mkdirs() else child.createNewFile()
            return "root/"+child.relativeTo(root).invariantSeparatorsPath
        }
        override fun openDocument(documentId: String, mode: String, signal: CancellationSignal?): ParcelFileDescriptor {
            if(mode.contains('w'))writes++
            if(failWrite && mode.contains('w')) { failWrite=false; throw FileNotFoundException("Provider write interrupted") }
            return ParcelFileDescriptor.open(file(documentId),ParcelFileDescriptor.parseMode(mode))
        }
        override fun isChildDocument(parentDocumentId: String, documentId: String)=documentId.startsWith("$parentDocumentId/")
    }
    private fun fixture(): Triple<Files,MarkdownFolder,ChatEngine> {
        val context: Context=RuntimeEnvironment.getApplication()
        val provider=Files().apply { root=File(context.cacheDir,"documents-${System.nanoTime()}").apply { mkdirs() } }
        provider.attachInfo(context,ProviderInfo().apply { authority="notechat.test.docs"; exported=true; grantUriPermissions=true; readPermission="android.permission.MANAGE_DOCUMENTS"; writePermission=readPermission })
        ShadowContentResolver.registerProviderInternal("notechat.test.docs",provider)
        val tree=DocumentsContract.buildTreeDocumentUri("notechat.test.docs","root").toString()
        val engine=ChatEngine(ChatState(listOf(Category("thoughts","Thoughts","Observations")),listOf(Message(id="one",text="First note",category="thoughts")),markdown=MarkdownState(tree)))
        return Triple(provider,MarkdownFolder(context,tree),engine)
    }
    @Test fun documentTreeSyncReadsEditorChangesAndResolvesLocalFileLinks() {
        val (provider,folder,engine)=fixture()
        folder.sync(engine,{}, { error("No attachments") })
        val name=MarkdownWorkspace.filename(engine.state,"thoughts")
        val file=File(provider.root,"Note Chat/$name")
        file.appendText("New paragraph from an editor\n")
        folder.sync(engine,{}, { error("No attachments") })
        assertEquals(2,engine.state.messages.size)
        assertEquals("New paragraph from an editor",engine.state.messages.last().text)
        assertNotNull(folder.link(name))
        assertNull(folder.link("../outside.md"))
        val stable=engine.state
        folder.sync(engine,{}, { error("No attachments") })
        assertEquals(stable,engine.state)
    }
    @Test fun interruptedWritesRetryWithoutDuplicatingAppendedMessages() {
        val (provider,folder,engine)=fixture()
        var saved=engine.state
        val persist: (ChatState)->Unit={ saved=StateJson.decode(StateJson.encode(it)) }
        folder.sync(engine,persist, { error("No attachments") })
        val name=MarkdownWorkspace.filename(engine.state,"thoughts")
        File(provider.root,"Note Chat/$name").appendText("Add this once\n")
        provider.failWrite=true
        assertThrows(Exception::class.java) { folder.sync(engine,persist, { error("No attachments") }) }
        assertEquals(2,saved.messages.size)
        val reopened=ChatEngine(saved)
        folder.sync(reopened,persist, { error("No attachments") })
        assertEquals(2,reopened.state.messages.size)
        assertEquals(1,reopened.state.messages.count { it.text=="Add this once" })
        assertTrue(reopened.state.markdown.pendingRendered.isEmpty())
    }
    @Test fun conflictingVersionsAreWrittenToASeparateFile() {
        val (provider,folder,engine)=fixture()
        folder.sync(engine,{}, { error("No attachments") })
        val name=MarkdownWorkspace.filename(engine.state,"thoughts")
        engine.edit("one","App edit")
        val file=File(provider.root,"Note Chat/$name")
        file.writeText(file.readText().replace("First note","Editor edit"))
        assertEquals(1,folder.sync(engine,{}, { error("No attachments") }))
        assertEquals("App edit",engine.state.messages.single().text)
        assertTrue(File(provider.root,"Note Chat").listFiles()!!.any { it.name.contains(".conflict-") && it.readText().contains("Editor edit") })
    }
    @Test fun selectingTheSameFolderRestoresRecordsAndLaterEditorChanges() {
        val (provider,folder,engine)=fixture()
        engine.approve("one")
        engine.addChat(Chat("journal","Journal"))
        engine.state=engine.state.copy(messages=engine.state.messages+Message(id="private",text="Private note",chatId="journal"))
        folder.sync(engine,{}, { error("No attachments") })
        val file=File(provider.root,"Note Chat/Thoughts.md")
        file.writeText(file.readText().replace("First note","Edited after uninstall"))
        val restored=folder.recover(engine.state.categories)!!
        assertEquals(engine.state.messages,restored.messages)
        assertEquals(engine.state.chats,restored.chats)
        val reopened=ChatEngine(restored)
        folder.sync(reopened,{}, { error("No attachments") })
        assertEquals("Edited after uninstall",reopened.state.messages.first().text)
        assertFalse(reopened.state.messages.first().approved)
        assertEquals("private",reopened.state.messages.last().id)
    }
    @Test fun oldFoldersRecoverWithoutWorkspaceMetadataAndObsoleteChatsDoNotBlockRecovery() {
        val (_,folder,engine)=fixture()
        engine.addChat(Chat("journal","Journal"))
        engine.state=engine.state.copy(messages=engine.state.messages+Message(id="private",text="Private note",chatId="journal"))
        folder.sync(engine,{}, { error("No attachments") })
        engine.promoteChat("journal","Personal observations")
        folder.sync(engine,{}, { error("No attachments") })
        val recovered=folder.recover(engine.state.categories)!!
        assertEquals(2,recovered.messages.size)
        assertTrue(recovered.chats.isEmpty())
        val old=FolderRecovery.reconstruct(mapOf("Thoughts.md" to MarkdownWorkspace.render(engine.state,"thoughts")),engine.state.categories)
        assertEquals("one",old.messages.single().id)
        assertFalse(old.messages.single().approved)
    }

    @Test fun unchangedFolderChecksDoNotRewriteFilesOrTheVaultAndPhotosAreRetained() {
        val (provider,folder,engine)=fixture()
        folder.sync(engine,{}, { error("No attachments") })
        provider.writes=0; var saves=0
        folder.sync(engine,{ saves++ }, { error("No attachments") })
        assertEquals(0,provider.writes); assertEquals(0,saves)
        val photo=Attachment(java.util.UUID.randomUUID().toString(),"avatar.jpg","image/jpeg",4)
        engine.state=engine.state.copy(categories=engine.state.categories.map { it.copy(photo=photo) })
        folder.sync(engine,{}, { byteArrayOf(1,2,3,4).inputStream() })
        assertTrue(File(provider.root,"Note Chat/${MarkdownWorkspace.mediaName(photo)}").exists())
        assertEquals(photo,folder.recover(engine.state.categories)!!.categories.single().photo)
    }

}
