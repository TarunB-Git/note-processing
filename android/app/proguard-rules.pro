-keep class ai.onnxruntime.** { *; }

-keep class org.vosk.** { *; }
-keep class com.sun.jna.** { *; }
-keepclasseswithmembernames class * { native <methods>; }
# JNA's desktop AWT helpers are unavailable and unused on Android.
-dontwarn java.awt.**
