package dev.notechat

import android.content.Context
import android.net.Uri
import android.provider.DocumentsContract
import java.io.InputStream

/** Access is limited to the directory selected through Android's document picker. */
class MarkdownFolder(private val context: Context, tree: String) {
    private val resolver = context.contentResolver
    private val root = Uri.parse(tree)
    private val directory = DocumentsContract.Document.MIME_TYPE_DIR
    private fun children(parent: Uri): Map<String,Uri> {
        val query = DocumentsContract.buildChildDocumentsUriUsingTree(root,DocumentsContract.getDocumentId(parent))
        return resolver.query(query,arrayOf(DocumentsContract.Document.COLUMN_DOCUMENT_ID,DocumentsContract.Document.COLUMN_DISPLAY_NAME),android.os.Bundle.EMPTY,null)?.use { cursor ->
            buildMap { while(cursor.moveToNext()) put(cursor.getString(1),DocumentsContract.buildDocumentUriUsingTree(root,cursor.getString(0))) }
        } ?: error("Cannot read the selected folder")
    }
    private fun child(parent: Uri, name: String, mime: String) = children(parent)[name]
        ?: DocumentsContract.createDocument(resolver,parent,mime,name) ?: error("Cannot create $name")
    private fun notes() = child(DocumentsContract.buildDocumentUriUsingTree(root,DocumentsContract.getTreeDocumentId(root)),"Note Chat",directory)
    private fun read(uri: Uri): String = resolver.openInputStream(uri)?.use { input ->
        val bytes=input.readBytesLimited(16*1024*1024); bytes.toString(Charsets.UTF_8).replace("\r\n","\n")
    } ?: error("Cannot read Markdown file")
    private fun size(uri: Uri): Long? = resolver.query(uri,arrayOf(DocumentsContract.Document.COLUMN_SIZE),android.os.Bundle.EMPTY,null)?.use {
        if(it.moveToFirst() && !it.isNull(0))it.getLong(0) else null
    }
    private fun write(parent: Uri, name: String, text: String) {
        val uri=child(parent,name,"text/markdown")
        resolver.openOutputStream(uri,"wt")?.use { it.write(text.toByteArray(Charsets.UTF_8)) } ?: error("Cannot write $name")
        check(read(uri)==text) { "The document provider did not save $name correctly" }
    }
    fun sync(engine: ChatEngine, vault: Vault, configure: () -> Unit = {}, predict: (Message) -> Suggestion = { Suggestion(null,"") }) = sync(engine,vault::save,vault::readMedia,configure,predict)
    fun sync(engine: ChatEngine, persist: (ChatState) -> Unit, readMedia: (Attachment) -> InputStream, configure: () -> Unit = {}, predict: (Message) -> Suggestion = { Suggestion(null,"") }): Int {
        val folder=notes()
        val state=engine.state
        val destinations=listOf<String?>(null)+state.categories.map { it.id }+state.chats.map { "chat:${it.id}" }
        val filenames=destinations.associateWith { route -> MarkdownWorkspace.filename(state,route?.takeUnless { it.startsWith("chat:") },route?.removePrefix("chat:")?.takeIf { route.startsWith("chat:") }) }
        val listing=children(folder)
        val external=filenames.values.mapNotNull { name -> listing[name]?.let { name to read(it) } }.toMap()
        val effective=external.mapValues { (name,source) ->
            if(state.markdown.pendingExternal[name]==source)state.markdown.pendingRendered[name] ?: source else source
        }
        val merged=MarkdownWorkspace.merge(state,effective)
        merged.conflicts.forEach { name -> write(folder,"${name.removeSuffix(".md")}.conflict-${System.currentTimeMillis()}.md",external.getValue(name)) }
        engine.state=merged.state
        configure()
        val known=state.messages.map { it.id }.toSet()
        engine.state=engine.state.copy(messages=engine.state.messages.map { message ->
            if(message.id !in known && message.category==null && message.trainable() && engine.state.preferences.classification) { val suggestion=predict(message); message.copy(category=suggestion.category,reason=suggestion.reason) } else message
        })
        val output=filenames.map { (category,name) -> name to MarkdownWorkspace.render(engine.state,category?.takeUnless { it.startsWith("chat:") },chatId=category?.removePrefix("chat:")?.takeIf { category.startsWith("chat:") }) }.toMap()
        val pending=output.filter { (name,text) -> text!=external[name] }
        engine.state=engine.state.copy(markdown=engine.state.markdown.copy(pendingExternal=external.filterKeys { it in pending },pendingRendered=pending))
        // Persist IDs before writing: retrying an interrupted sync must not duplicate editor additions.
        if(engine.state!=state || pending.isNotEmpty())persist(engine.state)
        val media=BackupArchive.attachments(engine.state)
        if(media.isNotEmpty()) {
            val assets=child(folder,"_attachments",directory); val existing=children(assets)
            media.forEach { attachment ->
                val name=MarkdownWorkspace.mediaName(attachment).substringAfter('/')
                if(name !in existing || existing[name]?.let(::size)?.let { it!=attachment.bytes } == true) {
                    val uri=child(assets,name,attachment.mime)
                    resolver.openOutputStream(uri,"wt")?.use { outputStream -> readMedia(attachment).use { it.copyTo(outputStream) } } ?: error("Cannot write attachment")
                }
            }
        }
        if(pending.isEmpty() && engine.state==state && state.markdown.pendingRendered.isEmpty()) {
            val metadata=children(folder)["workspace.json"]?.let(::read)
            val snapshot=StateJson.encode(BackupArchive.portable(engine.state)).toString(Charsets.UTF_8)
            if(metadata!=snapshot)write(folder,"workspace.json",snapshot)
            return merged.conflicts.size
        }
        output.forEach { (name,text) ->
            val latest=children(folder)[name]?.let(::read)
            check(latest==external[name]) { "$name changed during sync. Try again." }
            if(latest!=text)write(folder,name,text)
            if(engine.state.markdown.baseline[name]==text && name !in engine.state.markdown.pendingRendered)return@forEach
            engine.state=engine.state.copy(markdown=engine.state.markdown.copy(
                baseline=engine.state.markdown.baseline+(name to text),
                pendingExternal=engine.state.markdown.pendingExternal-name,pendingRendered=engine.state.markdown.pendingRendered-name))
            persist(engine.state)
        }
        engine.state=engine.state.copy(markdown=engine.state.markdown.copy(synced=System.currentTimeMillis()))
        persist(engine.state)
        write(folder,"workspace.json",StateJson.encode(BackupArchive.portable(engine.state)).toString(Charsets.UTF_8))
        return merged.conflicts.size
    }
    fun recover(templates: List<Category>, vault: Vault? = null): ChatState? {
        val folder=notes(); val listing=children(folder)
        val documents=listing.filterKeys { it.endsWith(".md") && !it.contains(".conflict-") }.mapValues { read(it.value) }
            .filterValues { MarkdownWorkspace.categoryId(it)!=null }
        val metadata=listing["workspace.json"]?.let { StateJson.decode(read(it).toByteArray(Charsets.UTF_8)) }
        if(metadata==null && documents.isEmpty())return null
        val assets=listing["_attachments"]?.let(::children) ?: emptyMap()
        val legacyMedia=if(metadata==null)assets.mapNotNull { (name,uri) ->
            val match=Regex("^([a-f0-9-]{36})-(.+)$").matchEntire(name) ?: return@mapNotNull null
            name to Attachment(match.groupValues[1],match.groupValues[2],resolver.getType(uri) ?: "application/octet-stream",size(uri) ?: 0)
        }.toMap() else emptyMap()
        var state=metadata ?: FolderRecovery.reconstruct(documents,templates,legacyMedia)
        BackupArchive.validate(state)
        if(vault!=null)BackupArchive.attachments(state).forEach { attachment ->
            val name=MarkdownWorkspace.mediaName(attachment).substringAfter('/')
            val uri=assets[name] ?: error("Folder backup is missing ${attachment.name}")
            if(!vault.hasMedia(attachment))resolver.openInputStream(uri)?.use { input ->
                val saved=vault.importMedia(input,attachment.name,attachment.mime,attachment.duration,attachment.file)
                require(saved.bytes==attachment.bytes) { "Folder attachment size differs from metadata" }
            } ?: error("Cannot restore attachment")
        }
        state=BackupArchive.portable(state).copy(markdown=MarkdownState(root.toString(),documents.filterValues { source -> (MarkdownWorkspace.categoryId(source)=="unassigned" || state.categories.any { it.id==MarkdownWorkspace.categoryId(source) }) && (MarkdownWorkspace.chatId(source)==null || state.chats.any { it.id==MarkdownWorkspace.chatId(source) }) }.mapValues { (_,source) ->
            MarkdownWorkspace.render(state,MarkdownWorkspace.categoryId(source)?.takeUnless { it=="unassigned" },chatId=MarkdownWorkspace.chatId(source))
        }))
        return state
    }
    fun link(target: String): Uri? {
        val name=target.substringBefore('#').removeSuffix(".md")+".md"
        val relative=name.replace('\\','/').split('/')
        if(relative.any { it==".." || it.isBlank() })return null
        val roots=listOf(notes(),DocumentsContract.buildDocumentUriUsingTree(root,DocumentsContract.getTreeDocumentId(root)))
        for(parent in roots) {
            var cursor: Uri?=parent
            relative.forEach { part -> cursor=cursor?.let { children(it)[part] } }
            if(cursor!=null)return cursor
        }
        return null
    }
    private fun InputStream.readBytesLimited(limit: Int): ByteArray {
        val output=java.io.ByteArrayOutputStream(); val buffer=ByteArray(8192)
        while(true) { val count=read(buffer); if(count<0)break; require(output.size()+count<=limit) { "Markdown file exceeds 16 MB" }; output.write(buffer,0,count) }
        return output.toByteArray()
    }
}
