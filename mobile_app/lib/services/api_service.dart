import 'dart:async';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:flutter_image_compress/flutter_image_compress.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;
import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

class ApiService {
  static const String _prefKeyUrl = 'api_base_url';
  static const String _prefKeyKey = 'api_key';
  static const Duration _timeout = Duration(seconds: 20);
  static const int _maxRetries = 2;
  static const _storage = FlutterSecureStorage();

  // ── Config ──────────────────────────────────────────────────────────────────

  static Future<String> getBaseUrl() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_prefKeyUrl) ??
        (kIsWeb ? 'http://127.0.0.1:8000' : 'http://10.0.2.2:8000');
  }

  static Future<void> setBaseUrl(String url) async {
    final trimmed = url.trim();
    final uri = Uri.tryParse(trimmed);
    if (uri == null || !uri.hasScheme || !['http', 'https'].contains(uri.scheme) || uri.host.isEmpty) {
      throw ArgumentError('Invalid URL: must start with http:// or https://');
    }
    // Release builds block cleartext traffic (see AndroidManifest), so fail early.
    if (kReleaseMode && uri.scheme != 'https') {
      throw ArgumentError('Release builds require an https:// URL');
    }
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_prefKeyUrl, trimmed.replaceAll(RegExp(r'/+$'), ''));
  }

  static Future<String?> getApiKey() async {
    return _storage.read(key: _prefKeyKey);
  }

  static Future<void> setApiKey(String key) async {
    await _storage.write(key: _prefKeyKey, value: key.trim());
  }

  // ── Predict ─────────────────────────────────────────────────────────────────

  /// Compress → upload → return raw JSON map.
  Future<Map<String, dynamic>> predict(File imageFile) async {
    final compressed = await _compress(imageFile);
    final baseUrl = await getBaseUrl();
    final apiKey = await getApiKey();
    final uri = Uri.parse('$baseUrl/predict');

    return _withRetry(() async {
      final request = http.MultipartRequest('POST', uri);
      if (apiKey != null && apiKey.isNotEmpty) {
        request.headers['X-API-Key'] = apiKey;
      }
      request.files.add(
        await http.MultipartFile.fromPath('file', compressed.path),
      );

      final streamed = await request.send().timeout(_timeout);
      final response = await http.Response.fromStream(streamed);

      if (response.statusCode == 200) {
        return json.decode(response.body) as Map<String, dynamic>;
      }
      throw ApiException(response.statusCode, response.body);
    });
  }

  // ── Health ──────────────────────────────────────────────────────────────────

  Future<bool> checkHealth() async {
    try {
      final baseUrl = await getBaseUrl();
      final response = await http
          .get(Uri.parse('$baseUrl/health'))
          .timeout(const Duration(seconds: 5));
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  // ── Helpers ─────────────────────────────────────────────────────────────────

  Future<XFile> _compress(File file) async {
    try {
      final dir = await getTemporaryDirectory();
      // Always a .jpg target: the plugin picks the output format from it.
      final target = p.join(dir.path, 'compressed_${DateTime.now().microsecondsSinceEpoch}.jpg');
      final result = await FlutterImageCompress.compressAndGetFile(
        file.absolute.path,
        target,
        quality: 82,
        minWidth: 640,
        minHeight: 640,
      );
      return result ?? XFile(file.path);
    } catch (e) {
      throw ImageProcessingException('Could not read this image ($e). Try another photo.');
    }
  }

  Future<T> _withRetry<T>(Future<T> Function() op) async {
    int attempt = 0;
    while (true) {
      try {
        return await op();
      } on ApiException catch (e) {
        // Only 503 (model loading) is safe to retry: a 500 on POST /predict may
        // already have logged a prediction server-side.
        if (e.statusCode != 503) rethrow;
        if (++attempt > _maxRetries) rethrow;
        await Future.delayed(Duration(seconds: attempt));
      } catch (e) {
        if (++attempt > _maxRetries) rethrow;
        await Future.delayed(Duration(seconds: attempt));
      }
    }
  }
}

/// The server returns HTTP 200 with {"error": "OBJECT_NOT_RECOGNIZED"} when the
/// image does not look like supported produce.
bool isObjectNotRecognized(Map<String, dynamic> json) {
  final err = json['error'];
  return err == 'OBJECT_NOT_RECOGNIZED' ||
      (err is Map && err['code'] == 'OBJECT_NOT_RECOGNIZED');
}

/// User-facing message for anything thrown while scanning.
String describeError(Object e) {
  if (e is ApiException) return e.toString();
  if (e is ImageProcessingException) return e.message;
  if (e is SocketException || e is TimeoutException || e is http.ClientException) {
    return 'Could not connect to server. Check the API URL in Settings.';
  }
  if (e is FormatException) return 'Unexpected response from server.';
  return 'Something went wrong: $e';
}

class ImageProcessingException implements Exception {
  final String message;
  ImageProcessingException(this.message);
  @override
  String toString() => message;
}

class ApiException implements Exception {
  final int statusCode;
  final String body;
  ApiException(this.statusCode, String rawBody)
      : body = rawBody.length > 200 ? rawBody.substring(0, 200) : rawBody;

  @override
  String toString() {
    if (statusCode == 413) return 'Image too large. Please use a smaller photo.';
    if (statusCode == 400) return 'Invalid image. Please try another photo.';
    if (statusCode == 503) return 'Model not ready. Please try again shortly.';
    if (statusCode == 401) return 'Invalid API key. Check your settings.';
    return 'Server error ($statusCode). Please try again.';
  }
}
