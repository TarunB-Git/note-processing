plugins { id("com.android.application"); id("org.jetbrains.kotlin.android") }

val signingFields = mapOf(
    "ANDROID_KEYSTORE_PATH" to "android.injected.signing.store.file",
    "ANDROID_KEYSTORE_PASSWORD" to "android.injected.signing.store.password",
    "ANDROID_KEY_ALIAS" to "android.injected.signing.key.alias",
    "ANDROID_KEY_PASSWORD" to "android.injected.signing.key.password"
)
val environmentSigning = signingFields.keys.associateWith { providers.environmentVariable(it).orNull }
val releaseSigning = if (environmentSigning.values.all { !it.isNullOrBlank() }) environmentSigning
    else signingFields.mapValues { providers.gradleProperty(it.value).orNull }
val hasReleaseSigning = releaseSigning.values.all { !it.isNullOrBlank() }

android {
    namespace = "dev.notechat"
    compileSdk = 35
    defaultConfig {
        applicationId = "dev.notechat"
        minSdk = 30
        targetSdk = 35
        versionCode = 8
        versionName = "1.5.0"
        testInstrumentationRunner = "dev.notechat.RuntimeChecks"
        ndk { abiFilters += if (providers.gradleProperty("emulator").orNull == "true") listOf("arm64-v8a", "x86_64") else listOf("arm64-v8a") }
    }
    signingConfigs {
        if (hasReleaseSigning) create("production") {
            storeFile = file(releaseSigning.getValue("ANDROID_KEYSTORE_PATH")!!)
            storePassword = releaseSigning.getValue("ANDROID_KEYSTORE_PASSWORD")
            keyAlias = releaseSigning.getValue("ANDROID_KEY_ALIAS")
            keyPassword = releaseSigning.getValue("ANDROID_KEY_PASSWORD")
        }
    }
    buildTypes {
        release {
            if (hasReleaseSigning) signingConfig = signingConfigs.getByName("production")
            isMinifyEnabled = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
        create("preview") {
            initWith(getByName("release"))
            signingConfig = signingConfigs.getByName("debug")
            matchingFallbacks += "release"
        }
    }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    kotlinOptions { jvmTarget = "17" }
    testOptions {
        unitTests.isReturnDefaultValues = true
        unitTests.isIncludeAndroidResources = true
        unitTests.all { it.systemProperty("robolectric.dependency.repo.url", "https://repo.maven.apache.org/maven2") }
    }
}

val verifyReleaseSigning by tasks.registering {
    doLast {
        check(hasReleaseSigning) { "Use Android Studio signing or provide: ${releaseSigning.keys.joinToString()}." }
        check(file(releaseSigning.getValue("ANDROID_KEYSTORE_PATH")!!).isFile) { "Release keystore does not exist." }
    }
}
tasks.matching { it.name == "preReleaseBuild" }.configureEach { dependsOn(verifyReleaseSigning) }

dependencies {
    implementation("com.google.mlkit:translate:17.0.3")
    implementation("com.google.android.gms:play-services-mlkit-language-id:17.0.0")
    implementation("com.google.android.gms:play-services-mlkit-text-recognition:19.0.1")
    implementation("com.alphacephei:vosk-android:0.3.75")
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.recyclerview:recyclerview:1.2.1")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.exifinterface:exifinterface:1.3.7")
    implementation("io.noties.markwon:core:4.6.2")
    implementation("io.noties.markwon:ext-strikethrough:4.6.2")
    implementation("io.noties.markwon:ext-tables:4.6.2")
    implementation("io.noties.markwon:ext-tasklist:4.6.2")
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")
    testImplementation("org.robolectric:robolectric:4.14.1")
}
