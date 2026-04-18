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
    if (uri == null || !uri.hasScheme || !['http', 'https'].contains(uri.scheme)) {
      throw ArgumentError('Invalid URL: must start with http:// or https://');
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
    final dir = await getTemporaryDirectory();
    final target = p.join(dir.path, 'compressed_${DateTime.now().microsecondsSinceEpoch}_${p.basename(file.path)}');
    final result = await FlutterImageCompress.compressAndGetFile(
      file.absolute.path,
      target,
      quality: 82,
      minWidth: 640,
      minHeight: 640,
    );
    return result ?? XFile(file.path);
  }

  Future<T> _withRetry<T>(Future<T> Function() op) async {
    int attempt = 0;
    while (true) {
      try {
        return await op();
      } on ApiException catch (e) {
        if (e.statusCode < 500) rethrow; // Don't retry 4xx
        if (++attempt > _maxRetries) rethrow;
        await Future.delayed(Duration(seconds: attempt));
      } catch (e) {
        if (++attempt > _maxRetries) rethrow;
        await Future.delayed(Duration(seconds: attempt));
      }
    }
  }
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
