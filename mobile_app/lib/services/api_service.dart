import 'dart:io' show Platform;
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:image_picker/image_picker.dart';

class ApiService {
  // Platform-aware base URL
  static String get baseUrl {
    if (kIsWeb) {
      return 'http://127.0.0.1:8000';
    } else {
      // Use PC local IP for physical devices on the same Wi-Fi
      // TODO: Configure via environment or app settings
      return 'http://10.0.2.2:8000'; // Android emulator localhost
    }
  } 

  Future<Map<String, dynamic>> predict(XFile imageFile) async {
    var uri = Uri.parse('$baseUrl/predict');
    var request = http.MultipartRequest('POST', uri);
    
    debugPrint("Sending request to $uri with file: ${imageFile.name}");

    // Add image file appropriately for Web vs Native
    if (kIsWeb) {
      final bytes = await imageFile.readAsBytes();
      request.files.add(
        http.MultipartFile.fromBytes(
          'file', 
          bytes,
          filename: imageFile.name.isNotEmpty ? imageFile.name : 'image.jpg',
        )
      );
    } else {
      request.files.add(
        await http.MultipartFile.fromPath(
          'file', 
          imageFile.path,
        )
      );
    }
    
    try {
      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);
      
      if (response.statusCode == 200) {
        debugPrint("Response received: ${response.body}");
        return json.decode(response.body);
      } else {
        debugPrint("Error: ${response.statusCode} - ${response.body}");
        throw Exception('Failed to predict: ${response.statusCode}');
      }
    } catch (e) {
      debugPrint("Exception: $e");
      throw Exception('Error connecting to API ($baseUrl): $e');
    }
  }
}

