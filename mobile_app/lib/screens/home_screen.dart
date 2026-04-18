import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import '../models/prediction_result.dart';
import '../services/api_service.dart';
import '../services/database_service.dart';
import '../widgets/result_card.dart';
import '../widgets/freshness_badge.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with AutomaticKeepAliveClientMixin {
  @override
  bool get wantKeepAlive => true;

  final _api = ApiService();
  final _picker = ImagePicker();

  File? _image;
  PredictionResult? _result;
  bool _loading = false;
  String? _error;

  Future<void> _pick(ImageSource source) async {
    final picked = await _picker.pickImage(source: source, imageQuality: 90);
    if (picked == null || !mounted) return;

    setState(() {
      _image = File(picked.path);
      _result = null;
      _error = null;
    });
    await _analyze(File(picked.path));
  }

  Future<void> _analyze(File file) async {
    setState(() { _loading = true; _error = null; });
    try {
      final json = await _api.predict(file);
      if (!mounted) return;
      final result = PredictionResult.fromJson(json, imagePath: file.path);
      await DatabaseService.insert(result);
      setState(() { _result = result; });
    } on ApiException catch (e) {
      if (mounted) setState(() { _error = e.toString(); });
    } catch (e) {
      if (mounted) setState(() { _error = 'Could not connect to server. Check your settings.'; });
    } finally {
      if (mounted) setState(() { _loading = false; });
    }
  }

  void _reset() => setState(() { _image = null; _result = null; _error = null; });

  @override
  Widget build(BuildContext context) {
    super.build(context);
    return Scaffold(
      appBar: AppBar(
        title: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 8, height: 8,
              decoration: const BoxDecoration(color: Color(0xFF00E676), shape: BoxShape.circle),
            ),
            const SizedBox(width: 8),
            const Text('FreshTrack AI'),
          ],
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings_outlined),
            onPressed: () => Navigator.pushNamed(context, '/settings'),
          ),
        ],
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(20, 8, 20, 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              _ImageArea(
                image: _image,
                loading: _loading,
                onReset: _reset,
              ),
              const SizedBox(height: 20),
              _ActionRow(
                onCamera: () => _pick(ImageSource.camera),
                onGallery: () => _pick(ImageSource.gallery),
                enabled: !_loading,
              ),
              const SizedBox(height: 28),
              if (_loading) const _LoadingState(),
              if (_error != null) _ErrorCard(message: _error!),
              if (_result != null && !_loading)
                ResultCard(result: _result!),
            ],
          ),
        ),
      ),
    );
  }
}

// ── Sub-widgets ───────────────────────────────────────────────────────────────

class _ImageArea extends StatelessWidget {
  final File? image;
  final bool loading;
  final VoidCallback onReset;

  const _ImageArea({required this.image, required this.loading, required this.onReset});

  @override
  Widget build(BuildContext context) {
    const border = Color(0xFF252D40);
    const primary = Color(0xFF00E676);

    return GestureDetector(
      onTap: image != null ? onReset : null,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 300),
        height: 280,
        decoration: BoxDecoration(
          color: const Color(0xFF131929),
          borderRadius: BorderRadius.circular(24),
          border: Border.all(
            color: image != null ? primary.withOpacity(0.4) : border,
            width: image != null ? 1.5 : 1,
          ),
        ),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(23),
          child: image != null
              ? Stack(fit: StackFit.expand, children: [
                  Image.file(image!, fit: BoxFit.cover),
                  if (loading)
                    Container(
                      color: Colors.black54,
                      child: const Center(
                        child: SpinKitRipple(color: Color(0xFF00E676), size: 60),
                      ),
                    ),
                  if (!loading)
                    Positioned(
                      top: 12, right: 12,
                      child: _IconChip(icon: Icons.close, label: 'Clear'),
                    ),
                ])
              : _EmptyState(),
        ),
      ),
    );
  }
}

class _EmptyState extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Container(
          width: 72, height: 72,
          decoration: BoxDecoration(
            color: const Color(0xFF00E676).withOpacity(0.08),
            shape: BoxShape.circle,
          ),
          child: const Icon(Icons.add_photo_alternate_outlined, color: Color(0xFF00E676), size: 32),
        ),
        const SizedBox(height: 16),
        const Text('Take or upload a photo', style: TextStyle(color: Colors.white70, fontSize: 15, fontWeight: FontWeight.w500)),
        const SizedBox(height: 6),
        Text('Supports JPG, PNG, WebP', style: TextStyle(color: Colors.white.withOpacity(0.3), fontSize: 12)),
      ],
    );
  }
}

class _IconChip extends StatelessWidget {
  final IconData icon;
  final String label;
  const _IconChip({required this.icon, required this.label});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.6),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white12),
      ),
      child: Row(mainAxisSize: MainAxisSize.min, children: [
        Icon(icon, size: 14, color: Colors.white70),
        const SizedBox(width: 4),
        Text(label, style: const TextStyle(color: Colors.white70, fontSize: 12)),
      ]),
    );
  }
}

class _ActionRow extends StatelessWidget {
  final VoidCallback onCamera, onGallery;
  final bool enabled;
  const _ActionRow({required this.onCamera, required this.onGallery, required this.enabled});

  @override
  Widget build(BuildContext context) {
    return Row(children: [
      Expanded(
        flex: 3,
        child: _PrimaryButton(
          icon: Icons.camera_alt_outlined,
          label: 'Take Photo',
          onTap: enabled ? onCamera : null,
        ),
      ),
      const SizedBox(width: 12),
      Expanded(
        child: _SecondaryButton(
          icon: Icons.photo_library_outlined,
          onTap: enabled ? onGallery : null,
        ),
      ),
    ]);
  }
}

class _PrimaryButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback? onTap;
  const _PrimaryButton({required this.icon, required this.label, this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedOpacity(
        opacity: onTap != null ? 1.0 : 0.4,
        duration: const Duration(milliseconds: 200),
        child: Container(
          height: 56,
          decoration: BoxDecoration(
            gradient: const LinearGradient(
              colors: [Color(0xFF00E676), Color(0xFF00BFA5)],
            ),
            borderRadius: BorderRadius.circular(16),
          ),
          child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
            Icon(icon, color: const Color(0xFF001A0D), size: 20),
            const SizedBox(width: 8),
            Text(label, style: const TextStyle(color: Color(0xFF001A0D), fontWeight: FontWeight.w700, fontSize: 15)),
          ]),
        ),
      ),
    );
  }
}

class _SecondaryButton extends StatelessWidget {
  final IconData icon;
  final VoidCallback? onTap;
  const _SecondaryButton({required this.icon, this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedOpacity(
        opacity: onTap != null ? 1.0 : 0.4,
        duration: const Duration(milliseconds: 200),
        child: Container(
          height: 56,
          decoration: BoxDecoration(
            color: const Color(0xFF1C2333),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: const Color(0xFF252D40)),
          ),
          child: Icon(icon, color: Colors.white70, size: 22),
        ),
      ),
    );
  }
}

class _LoadingState extends StatelessWidget {
  const _LoadingState();

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(28),
      decoration: BoxDecoration(
        color: const Color(0xFF1C2333),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: const Color(0xFF252D40)),
      ),
      child: Column(children: [
        const SpinKitThreeBounce(color: Color(0xFF00E676), size: 28),
        const SizedBox(height: 16),
        Text('Analysing fruit...', style: TextStyle(color: Colors.white.withOpacity(0.6), fontSize: 14)),
      ]),
    );
  }
}

class _ErrorCard extends StatelessWidget {
  final String message;
  const _ErrorCard({required this.message});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF2A1A1A),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.red.withOpacity(0.3)),
      ),
      child: Row(children: [
        const Icon(Icons.error_outline, color: Colors.redAccent, size: 20),
        const SizedBox(width: 12),
        Expanded(child: Text(message, style: const TextStyle(color: Colors.redAccent, fontSize: 13))),
      ]),
    );
  }
}
