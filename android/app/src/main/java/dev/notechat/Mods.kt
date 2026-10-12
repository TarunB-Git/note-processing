package dev.notechat

import android.content.Context
import android.graphics.BitmapFactory
import com.google.android.gms.common.moduleinstall.ModuleInstall
import com.google.android.gms.common.moduleinstall.ModuleInstallRequest
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.common.model.DownloadConditions
import com.google.mlkit.common.model.RemoteModelManager
import com.google.mlkit.nl.languageid.LanguageIdentification
import com.google.mlkit.nl.translate.TranslateLanguage
import com.google.mlkit.nl.translate.TranslateRemoteModel
import com.google.mlkit.nl.translate.Translation
import com.google.mlkit.nl.translate.TranslatorOptions
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import java.io.File
import java.util.concurrent.TimeUnit

/** Invoked only after a mod is enabled; text and images are processed on-device. */
object Mods {
    fun languages()=TranslateLanguage.getAllLanguages().sortedBy { java.util.Locale.forLanguageTag(it).displayLanguage }
    fun downloaded()=Tasks.await(RemoteModelManager.getInstance().getDownloadedModels(TranslateRemoteModel::class.java),30,TimeUnit.SECONDS).map { it.language }.toSet()
    fun download(language: String) {
        if(language==TranslateLanguage.ENGLISH)return
        Tasks.await(RemoteModelManager.getInstance().download(TranslateRemoteModel.Builder(language).build(),DownloadConditions.Builder().requireWifi().build()),5,TimeUnit.MINUTES)
    }
    fun remove(language: String) { Tasks.await(RemoteModelManager.getInstance().deleteDownloadedModel(TranslateRemoteModel.Builder(language).build()),30,TimeUnit.SECONDS) }
    fun translate(context: Context,text: String,target: String): String {
        val identifier=LanguageIdentification.getClient()
        val tag=try {
            Tasks.await(ModuleInstall.getClient(context).installModules(ModuleInstallRequest.newBuilder().addApi(identifier).build()),5,TimeUnit.MINUTES)
            Tasks.await(identifier.identifyLanguage(text),30,TimeUnit.SECONDS)
        } finally { identifier.close() }
        val source=TranslateLanguage.fromLanguageTag(tag) ?: error("Language could not be identified")
        if(source==target)return text
        val translator=Translation.getClient(TranslatorOptions.Builder().setSourceLanguage(source).setTargetLanguage(target).build())
        return try {
            Tasks.await(translator.downloadModelIfNeeded(DownloadConditions.Builder().requireWifi().build()),5,TimeUnit.MINUTES)
            Tasks.await(translator.translate(text),30,TimeUnit.SECONDS)
        } finally { translator.close() }
    }
    fun installOcr(context: Context) {
        val recognizer=TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
        try { Tasks.await(ModuleInstall.getClient(context).installModules(ModuleInstallRequest.newBuilder().addApi(recognizer).build()),5,TimeUnit.MINUTES) }
        finally { recognizer.close() }
    }
    fun ocr(file: File): String {
        val bounds=BitmapFactory.Options().apply { inJustDecodeBounds=true }; BitmapFactory.decodeFile(file.path,bounds)
        var sample=1; while(maxOf(bounds.outWidth,bounds.outHeight)/sample>2048)sample*=2
        val bitmap=BitmapFactory.decodeFile(file.path,BitmapFactory.Options().apply { inSampleSize=sample }) ?: error("Cannot read this image")
        val rotation=when(androidx.exifinterface.media.ExifInterface(file).getAttributeInt(androidx.exifinterface.media.ExifInterface.TAG_ORIENTATION,1)) { 3->180; 6->90; 8->270; else->0 }
        val recognizer=TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
        return try { Tasks.await(recognizer.process(InputImage.fromBitmap(bitmap,rotation)),30,TimeUnit.SECONDS).text }
        finally { recognizer.close(); bitmap.recycle() }
    }
}
