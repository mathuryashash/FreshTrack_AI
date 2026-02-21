import 'dart:io' show File;
import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:image_picker/image_picker.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:percent_indicator/circular_percent_indicator.dart';
import 'package:camera/camera.dart';
import '../services/api_service.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with SingleTickerProviderStateMixin {
  bool _isScanning = false;
  XFile? _imageFile;
  Map<String, dynamic>? _result;
  String _cameraError = '';
  final ApiService _apiService = ApiService();

  CameraController? _cameraController;
  List<CameraDescription>? _cameras;
  bool _isCameraInitialized = false;
  Timer? _scanTimer;

  // Result smoothing: keep last 3 raw results and update UI only when 2/3 agree
  final List<Map<String, dynamic>> _resultBuffer = [];
  static const int _bufferSize = 3;
  static const double _confidenceThreshold = 0.60;
  
  late AnimationController _animationController;
  late Animation<double> _animation;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    
    // Setup scanning line animation
    _animationController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat(reverse: true);
    
    _animation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _animationController, curve: Curves.easeInOut),
    );
  }

  Future<void> _initializeCamera() async {
    try {
      if (!mounted) return;
      setState(() => _cameraError = "Getting cameras...");
      
      _cameras = await availableCameras().timeout(const Duration(seconds: 5));
      if (_cameras != null && _cameras!.isNotEmpty) {
        setState(() => _cameraError = "Initializing controller...");
        _cameraController = CameraController(
          _cameras![0], 
          ResolutionPreset.medium,
          enableAudio: false,
        );

        await _cameraController!.initialize().timeout(const Duration(seconds: 10));
        
        if (mounted) {
          setState(() {
            _isCameraInitialized = true;
            _cameraError = '';
          });
          _startAutoScan();
        }
      } else {
        setState(() {
           _cameraError = "No cameras returned from browser.";
        });
      }
    } catch (e) {
      debugPrint("Error initializing camera: $e");
      if (mounted) {
        setState(() {
           _cameraError = "Camera Error: $e";
        });
      }
    }
  }

  void _startAutoScan() {
    // Scan every 5 seconds to reduce noise (was 3s)
    _scanTimer = Timer.periodic(const Duration(seconds: 5), (timer) async {
      if (!mounted || !_isCameraInitialized || _cameraController == null) return;
      if (_isScanning) return;
      if (_imageFile != null) return;

      try {
        final XFile picture = await _cameraController!.takePicture();
        _analyzeImage(picture, isAutoScan: true);
      } catch (e) {
        debugPrint("Auto-scan error: $e");
      }
    });
  }

  /// Only update the displayed result when the rolling buffer has a majority.
  void _updateResultBuffer(Map<String, dynamic> newResult) {
    final double confidence = (newResult['freshness_confidence'] as num?)?.toDouble() ?? 0.0;
    
    // Skip low-confidence results entirely
    if (confidence < _confidenceThreshold) {
      debugPrint("Skipping low-confidence result: ${(confidence * 100).toStringAsFixed(0)}%");
      return;
    }

    _resultBuffer.add(newResult);
    if (_resultBuffer.length > _bufferSize) {
      _resultBuffer.removeAt(0);
    }

    // Count votes for each freshness label in the buffer
    final Map<String, int> votes = {};
    for (final r in _resultBuffer) {
      final label = r['freshness'] as String? ?? 'Unknown';
      votes[label] = (votes[label] ?? 0) + 1;
    }

    // Find the majority winner (needs > half the buffer)
    final int majority = (_bufferSize / 2).ceil();
    String? winner;
    int maxVotes = 0;
    votes.forEach((label, count) {
      if (count > maxVotes) {
        maxVotes = count;
        winner = label;
      }
    });

    if (winner != null && maxVotes >= majority) {
      // Use the most recent result that matches the winner for full details
      final winnerResult = _resultBuffer.lastWhere(
        (r) => (r['freshness'] as String?) == winner,
        orElse: () => newResult,
      );
      if (mounted) {
        setState(() {
          _result = winnerResult;
        });
      }
    }
  }

  @override
  void dispose() {
    _scanTimer?.cancel();
    _cameraController?.dispose();
    _animationController.dispose();
    super.dispose();
  }

  Future<void> _pickImage(ImageSource source) async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: source);

    if (pickedFile != null) {
      setState(() {
        _imageFile = pickedFile;
        _result = null; 
      });
      _analyzeImage(pickedFile, isAutoScan: false);
    }
  }

  Future<void> _analyzeImage(XFile imageFile, {required bool isAutoScan}) async {
    setState(() {
      _isScanning = true;
    });

    try {
      final result = await _apiService.predict(imageFile);
      if (!mounted) return;
      if (isAutoScan && _imageFile != null) return;

      if (isAutoScan) {
        // Use smoothing for auto-scan results
        _updateResultBuffer(result);
      } else {
        // Manual scan: show result directly (user intentionally triggered it)
        _resultBuffer.clear();
        setState(() {
          _result = result;
        });
      }
    } catch (e) {
      debugPrint("Predict Error: $e");
      if (!isAutoScan && mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Error: $e'), 
            backgroundColor: Colors.redAccent,
          ),
        );
      }
    } finally {
      if (mounted) {
        setState(() {
          _isScanning = false;
        });
      }
    }
  }

  Color _getStatusColor(String freshness) {
    if (freshness.toLowerCase().contains('fresh') && !freshness.toLowerCase().contains('not')) {
      return const Color(0xFF2BEE7C); // Emerald Green
    } else if (freshness.toLowerCase().contains('overripe') || freshness.toLowerCase().contains('spoiled') || freshness.toLowerCase().contains('rotten')) {
      return const Color(0xFFFF5252); // Red
    } else if (freshness.toLowerCase().contains('semi')) {
       return const Color(0xFFB2FF59);
    }
    return const Color(0xFF00B4D8); // Blue default
  }

  Widget _buildCameraViewfinder() {
    if (_imageFile != null) {
      // Show the picked gallery image
      return Container(
        height: 320,
        width: double.infinity,
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(24),
          border: Border.all(
            color: const Color(0xFF2BEE7C).withOpacity(0.5),
            width: 2,
          ),
        ),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(22),
          child: kIsWeb 
            ? Image.network(_imageFile!.path, fit: BoxFit.cover)
            : Image.file(File(_imageFile!.path), fit: BoxFit.cover),
        ),
      );
    }

    if (_cameraError.isNotEmpty) {
      return Container(
        height: 320,
        width: double.infinity,
        decoration: BoxDecoration(
          color: const Color(0xFF1A1D24),
          borderRadius: BorderRadius.circular(24),
          border: Border.all(color: const Color(0xFFFF5252), width: 2),
        ),
        child: Center(
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Text(
              _cameraError,
              textAlign: TextAlign.center,
              style: const TextStyle(color: Colors.redAccent),
            ),
          ),
        ),
      );
    }

    if (!_isCameraInitialized || _cameraController == null) {
      return Container(
        height: 320,
        width: double.infinity,
        decoration: BoxDecoration(
          color: const Color(0xFF1A1D24),
          borderRadius: BorderRadius.circular(24),
          border: Border.all(
            color: const Color(0xFF2E3340),
            width: 2,
          ),
        ),
        child: const Center(
          child: SpinKitRing(color: Color(0xFF2BEE7C), size: 40.0, lineWidth: 3.0),
        ),
      );
    }

    // Active Camera Viewfinder
    return Container(
      height: 320,
      width: double.infinity,
      decoration: BoxDecoration(
        color: Colors.black,
        borderRadius: BorderRadius.circular(24),
        border: Border.all(
          color: const Color(0xFF2BEE7C).withOpacity(0.3),
          width: 2,
        ),
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(22),
        child: Stack(
          fit: StackFit.expand,
          children: [
            // Live Feed
            FittedBox(
              fit: BoxFit.cover,
              child: SizedBox(
                width: _cameraController!.value.previewSize?.height ?? 1,
                height: _cameraController!.value.previewSize?.width ?? 1,
                child: CameraPreview(_cameraController!),
              ),
            ),
            Container(color: Colors.black.withOpacity(0.1)),
            // Animated Scanning Line
            AnimatedBuilder(
              animation: _animation,
              builder: (context, child) {
                return Positioned(
                  top: _animation.value * 300,
                  left: 0,
                  right: 0,
                  child: Container(
                    height: 2,
                    decoration: BoxDecoration(
                      boxShadow: [
                        BoxShadow(
                          color: const Color(0xFF2BEE7C).withOpacity(0.8),
                          blurRadius: 10,
                          spreadRadius: 2,
                        ),
                      ],
                    ),
                    child: Container(color: const Color(0xFF2BEE7C)),
                  ),
                );
              },
            ),
            Positioned(top: 20, left: 20, child: _buildCorner(isTop: true, isLeft: true)),
            Positioned(top: 20, right: 20, child: _buildCorner(isTop: true, isLeft: false)),
            Positioned(bottom: 20, left: 20, child: _buildCorner(isTop: false, isLeft: true)),
            Positioned(bottom: 20, right: 20, child: _buildCorner(isTop: false, isLeft: false)),
            
            // Scanning Indicator
            if (_isScanning)
              Positioned(
                bottom: 16,
                right: 16,
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  decoration: BoxDecoration(
                    color: Colors.black54,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: const Row(
                    children: [
                      SizedBox(
                        width: 12, height: 12,
                        child: CircularProgressIndicator(strokeWidth: 2, color: Color(0xFF2BEE7C)),
                      ),
                      SizedBox(width: 8),
                      Text("Analyzing...", style: TextStyle(color: Colors.white, fontSize: 10)),
                    ],
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildCorner({required bool isTop, required bool isLeft}) {
    return Container(
      width: 30,
      height: 30,
      decoration: BoxDecoration(
        border: Border(
          top: isTop ? const BorderSide(color: Color(0xFF2BEE7C), width: 3) : BorderSide.none,
          bottom: !isTop ? const BorderSide(color: Color(0xFF2BEE7C), width: 3) : BorderSide.none,
          left: isLeft ? const BorderSide(color: Color(0xFF2BEE7C), width: 3) : BorderSide.none,
          right: !isLeft ? const BorderSide(color: Color(0xFF2BEE7C), width: 3) : BorderSide.none,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0F1115), 
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: const Text(
          'FreshTrack AI',
          style: TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.w600,
            letterSpacing: 0.5,
          ),
        ),
        centerTitle: true,
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Live Camera Viewfinder or Captured Image
              GestureDetector(
                onTap: () {
                  if (_imageFile != null) {
                     setState(() {
                       _imageFile = null;
                       _result = null;
                     });
                  }
                },
                child: _buildCameraViewfinder(),
              ),

              const SizedBox(height: 24),

              // Action Buttons
              Row(
                children: [
                  Expanded(
                    flex: 4,
                    child: Container(
                      height: 60,
                      decoration: BoxDecoration(
                        color: const Color(0xFF2E3340),
                        borderRadius: BorderRadius.circular(16),
                      ),
                      child: ElevatedButton(
                        onPressed: () {
                          // Manually force a scan
                          if (_isCameraInitialized && _cameraController != null) {
                            _cameraController!.takePicture().then((picture) {
                               _analyzeImage(picture, isAutoScan: false);
                            });
                          }
                        },
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.transparent,
                          shadowColor: Colors.transparent,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(16),
                          ),
                        ),
                        child: const Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(Icons.document_scanner, color: Colors.white),
                            SizedBox(width: 8),
                            Text(
                              'Force Scan Now',
                              style: TextStyle(
                                color: Colors.white,
                                fontSize: 16,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    flex: 1,
                    child: SizedBox(
                      height: 60,
                      child: ElevatedButton(
                        onPressed: () => _pickImage(ImageSource.gallery),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFF2E3340),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(16),
                          ),
                        ),
                        child: const Icon(Icons.photo_library, color: Colors.white),
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 32),

              // Analysis Section
              if (_result != null)
                _buildAnalysisCard()
              else if (_isScanning && _result == null)
                const Center(
                   child: Padding(
                     padding: EdgeInsets.only(top: 20),
                     child: Text("Detecting produce...", style: TextStyle(color: Colors.grey)),
                   )
                ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildAnalysisCard() {
    String freshness = _result?['freshness'] ?? 'Unknown';
    double confidence = ((_result?['freshness_confidence'] ?? 0.0) as num).toDouble();
    String quality = _result?['quality'] ?? 'Unknown';
    double shelfLife = ((_result?['shelf_life_days'] ?? 0.0) as num).toDouble();
    Color statusColor = _getStatusColor(freshness);

    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: const Color(0xFF1E222A), // Charcoal card background
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: const Color(0xFF2E3340)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                'Live Analysis',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  color: statusColor.withOpacity(0.15),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  '${(confidence * 100).toStringAsFixed(0)}% Match',
                  style: TextStyle(
                    color: statusColor,
                    fontWeight: FontWeight.bold,
                    fontSize: 12,
                  ),
                ),
              )
            ],
          ),
          const SizedBox(height: 24),
          Row(
            children: [
              CircularPercentIndicator(
                radius: 40.0,
                lineWidth: 8.0,
                percent: confidence,
                center: Text(
                  "${(confidence * 100).toStringAsFixed(0)}%",
                  style: const TextStyle(fontWeight: FontWeight.bold, color: Colors.white, fontSize: 18),
                ),
                progressColor: statusColor,
                backgroundColor: const Color(0xFF2E3340),
                circularStrokeCap: CircularStrokeCap.round,
              ),
              const SizedBox(width: 24),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _buildResultRow('Freshness', freshness, statusColor),
                    const SizedBox(height: 12),
                    _buildResultRow('Condition', quality, Colors.white70),
                    const SizedBox(height: 12),
                    _buildResultRow('Shelf Life', '${shelfLife.toStringAsFixed(1)} Days', Colors.white70),
                  ],
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildResultRow(String label, String value, Color valueColor) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(
          label,
          style: TextStyle(
            color: Colors.grey[500],
            fontSize: 14,
            fontWeight: FontWeight.w500,
          ),
        ),
        Text(
          value,
          style: TextStyle(
            color: valueColor,
            fontSize: 16,
            fontWeight: FontWeight.w600,
          ),
        ),
      ],
    );
  }
}

