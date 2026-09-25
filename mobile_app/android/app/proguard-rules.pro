# ONNX Runtime's JNI layer looks up Java classes/methods by name; R8 must not
# rename or strip them (flutter_onnxruntime docs, "mid == null" crash).
-keep class ai.onnxruntime.** { *; }
