import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:path_provider/path_provider.dart';
import '../services/classifier.dart';
import '../services/database_service.dart';
import '../services/pipeline.dart';
import '../widgets/result_card.dart';
import '../widgets/scan_view.dart';
import 'result_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with AutomaticKeepAliveClientMixin {
  @override
  bool get wantKeepAlive => true;

  final _picker = ImagePicker();

  File? _image;
  Analysis? _analysis;
  List<ScanItem> _items = const []; // detections, plus boxes the user draws
  bool _loading = false;
  bool _selecting = false; // drawing a box by hand
  String? _error;
  int _gen = 0; // bumped per scan and on Clear; results of an older scan are dropped

  Future<void> _pick(ImageSource source) async {
    // Warm the model while the picker is open; errors resurface in _analyze.
    Classifier.instance.load().ignore();
    // image_picker handles both sources via the system camera / photo picker,
    // so there is no custom camera screen (and no CAMERA permission) to fail.
    final XFile? picked;
    try {
      picked = await _picker.pickImage(
        source: source,
        maxWidth: 1600,
        maxHeight: 1600,
        imageQuality: 90,
      );
    } catch (e) {
      if (mounted) {
        setState(() => _error = source == ImageSource.camera
            ? 'No camera available on this device.'
            : 'Could not open the gallery: $e');
      }
      return;
    }
    if (picked == null || !mounted) return;
    final file = File(picked.path);
    setState(() => _image = file);
    await _analyze(file);
  }

  Future<void> _analyze(File file) async {
    final gen = ++_gen;
    setState(() {
      _loading = true;
      _error = null;
      _analysis = null;
      _items = const [];
      _selecting = false;
    });
    try {
      final a = await Classifier.instance.analyse(file.path);
      if (!mounted || gen != _gen) return;
      // Show results first; saving the crops to history can follow.
      setState(() {
        _analysis = a;
        _items = a.items;
        _loading = false;
      });
      await _save(file.path, a.accepted);
    } catch (e) {
      if (mounted && gen == _gen) setState(() => _error = describeScanError(e));
    } finally {
      if (mounted && gen == _gen && _loading) setState(() => _loading = false);
    }
  }

  /// One history row per recognised item, with the item's crop as its picture.
  Future<void> _save(String imagePath, List<ScanItem> items) async {
    if (items.isEmpty) return;
    final dir = (await getTemporaryDirectory()).path;
    final ts = DateTime.now().microsecondsSinceEpoch;
    final thumbs = await Classifier.instance.saveCrops(imagePath, items, dir, 'crop_$ts');
    for (var k = 0; k < items.length; k++) {
      await DatabaseService.insert(items[k].classification.toResult(imagePath: thumbs[k], id: '${ts}_$k'));
    }
  }

  Future<void> _onSelected(Detection box) async {
    final file = _image;
    if (file == null) return;
    final gen = _gen;
    setState(() {
      _selecting = false;
      _loading = true;
      _error = null;
    });
    try {
      final item = await Classifier.instance.classifyRegion(file.path, box);
      if (!mounted || gen != _gen) return;
      setState(() {
        // A rejected whole-photo guess is superseded by the user's own box.
        _items = [for (final i in _items) if (i.box != null || i.classification.isProduce) i, item];
        _loading = false;
      });
      if (!item.classification.isProduce) {
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
          content: Text("Couldn't recognise that area. Try a tighter box around one item."),
        ));
      } else {
        await _save(file.path, [item]);
      }
    } catch (e) {
      if (mounted && gen == _gen) setState(() => _error = describeScanError(e));
    } finally {
      if (mounted && gen == _gen && _loading) setState(() => _loading = false);
    }
  }

  void _open(ScanItem item) => Navigator.push(
        context,
        MaterialPageRoute(builder: (_) => ResultScreen(result: item.classification.toResult())),
      );

  void _startSelecting() => setState(() => _selecting = true);

  void _reset() => setState(() {
        _gen++;
        _image = null;
        _analysis = null;
        _items = const [];
        _error = null;
        _selecting = false;
      });

  @override
  Widget build(BuildContext context) {
    super.build(context);
    final a = _analysis;
    final accepted = [for (final i in _items) if (i.classification.isProduce) i];
    final rejected = _items.length - accepted.length;
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
            icon: const Icon(Icons.info_outline),
            tooltip: 'About',
            onPressed: () => Navigator.pushNamed(context, '/settings'),
          ),
        ],
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          // While drawing a box, drags belong to the photo, not the page.
          physics: _selecting ? const NeverScrollableScrollPhysics() : null,
          padding: const EdgeInsets.fromLTRB(20, 8, 20, 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              if (_selecting) _SelectBanner(onCancel: () => setState(() => _selecting = false)),
              if (a != null && _image != null)
                _AnalysedImage(
                  image: _image!,
                  analysis: a,
                  items: _items,
                  selecting: _selecting,
                  loading: _loading,
                  onReset: _reset,
                  onTapItem: (k) {
                    final c = _items[k].classification;
                    if (c.isProduce) return _open(_items[k]);
                    final guess = c.toResult().produceLabel ?? 'unknown';
                    ScaffoldMessenger.of(context)
                      ..hideCurrentSnackBar()
                      ..showSnackBar(SnackBar(
                        content: Text('Not recognised. Closest match: $guess, but not confident enough. '
                            'Try "Select area" with a tighter box, or a closer photo.'),
                      ));
                  },
                  onSelected: _onSelected,
                )
              else
                _ImageArea(
                  image: _image,
                  loading: _loading,
                  onReset: _reset,
                ),
              const SizedBox(height: 20),
              _ActionRow(
                onCamera: () => _pick(ImageSource.camera),
                onGallery: () => _pick(ImageSource.gallery),
                enabled: !_loading && !_selecting,
              ),
              const SizedBox(height: 28),
              if (_loading && a == null) const _LoadingState(),
              if (_error != null) _ErrorCard(message: _error!),
              if (a != null && !_selecting) ...[
                if (accepted.isEmpty)
                  Center(child: _OodPopup(onRetry: _reset, onSelect: _startSelecting))
                else ...[
                  _ResultsHeader(
                    count: accepted.length,
                    wholePhoto: !a.detected && _items.every((i) => !i.manual),
                    onSelect: _loading ? null : _startSelecting,
                    onClear: _reset,
                  ),
                  const SizedBox(height: 12),
                  if (accepted.length == 1)
                    GestureDetector(
                      onTap: () => _open(accepted.first),
                      child: ResultCard(result: accepted.first.classification.toResult()),
                    )
                  else
                    for (final (k, item) in accepted.indexed)
                      Padding(
                        padding: const EdgeInsets.only(bottom: 10),
                        child: ItemTile(number: k + 1, item: item, onTap: () => _open(item)),
                      ),
                  if (rejected > 0)
                    Padding(
                      padding: const EdgeInsets.only(top: 6),
                      child: Text(
                        '$rejected more ${rejected == 1 ? 'thing was' : 'things were'} not recognised '
                        '(grey boxes). Use "Select area" to try one again.',
                        style: TextStyle(color: Colors.white.withValues(alpha: 0.5), fontSize: 12.5),
                      ),
                    ),
                ],
              ],
            ],
          ),
        ),
      ),
    );
  }
}

// ── Sub-widgets ───────────────────────────────────────────────────────────────

/// The analysed photo with its boxes; replaces the plain preview after a scan.
class _AnalysedImage extends StatelessWidget {
  final File image;
  final Analysis analysis;
  final List<ScanItem> items;
  final bool selecting, loading;
  final VoidCallback onReset;
  final ValueChanged<int> onTapItem;
  final ValueChanged<Detection> onSelected;

  const _AnalysedImage({
    required this.image,
    required this.analysis,
    required this.items,
    required this.selecting,
    required this.loading,
    required this.onReset,
    required this.onTapItem,
    required this.onSelected,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFF131929),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(
          color: selecting ? Colors.white70 : const Color(0xFF00E676).withValues(alpha: 0.4),
          width: 1.5,
        ),
      ),
      clipBehavior: Clip.antiAlias,
      child: Stack(children: [
        ScanImage(
          image: image,
          imageWidth: analysis.width,
          imageHeight: analysis.height,
          items: items,
          selecting: selecting && !loading,
          onTapItem: onTapItem,
          onSelected: onSelected,
        ),
        if (loading)
          const Positioned.fill(
            child: ColoredBox(
              color: Colors.black54,
              child: Center(child: SpinKitRipple(color: Color(0xFF00E676), size: 60)),
            ),
          ),
      ]),
    );
  }
}

class _ResultsHeader extends StatelessWidget {
  final int count;
  final bool wholePhoto;
  final VoidCallback? onSelect;
  final VoidCallback onClear;
  const _ResultsHeader({required this.count, required this.wholePhoto, required this.onSelect, required this.onClear});

  @override
  Widget build(BuildContext context) {
    final title = wholePhoto ? 'Whole photo' : (count == 1 ? '1 item found' : '$count items found');
    return Row(children: [
      Expanded(
        child: Text(title, style: const TextStyle(color: Colors.white, fontSize: 17, fontWeight: FontWeight.w700)),
      ),
      TextButton.icon(
        onPressed: onSelect,
        icon: const Icon(Icons.crop_free, size: 18),
        label: const Text('Select area'),
      ),
      IconButton(
        onPressed: onClear,
        tooltip: 'Clear',
        icon: const Icon(Icons.close, color: Colors.white54),
      ),
    ]);
  }
}

class _SelectBanner extends StatelessWidget {
  final VoidCallback onCancel;
  const _SelectBanner({required this.onCancel});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.fromLTRB(14, 6, 6, 6),
      decoration: BoxDecoration(
        color: const Color(0xFF1C2333),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.white24),
      ),
      child: Row(children: [
        const Icon(Icons.crop_free, color: Colors.white70, size: 20),
        const SizedBox(width: 10),
        const Expanded(
          child: Text('Drag a box around one fruit or vegetable',
              style: TextStyle(color: Colors.white, fontSize: 14)),
        ),
        TextButton(onPressed: onCancel, child: const Text('Cancel')),
      ]),
    );
  }
}

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
            color: image != null ? primary.withValues(alpha: 0.4) : border,
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
            color: const Color(0xFF00E676).withValues(alpha: 0.08),
            shape: BoxShape.circle,
          ),
          child: const Icon(Icons.add_photo_alternate_outlined, color: Color(0xFF00E676), size: 32),
        ),
        const SizedBox(height: 16),
        const Text('Take or upload a photo', style: TextStyle(color: Colors.white70, fontSize: 15, fontWeight: FontWeight.w500)),
        const SizedBox(height: 6),
        Text('Supports JPG, PNG, WebP', style: TextStyle(color: Colors.white.withValues(alpha: 0.3), fontSize: 12)),
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
        color: Colors.black.withValues(alpha: 0.6),
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
    return Semantics(
      button: true,
      enabled: onTap != null,
      label: 'Choose a photo from the gallery',
      child: GestureDetector(
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
        Text('Analysing fruit...', style: TextStyle(color: Colors.white.withValues(alpha: 0.6), fontSize: 14)),
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
        border: Border.all(color: Colors.red.withValues(alpha: 0.3)),
      ),
      child: Row(children: [
        const Icon(Icons.error_outline, color: Colors.redAccent, size: 20),
        const SizedBox(width: 12),
        Expanded(child: Text(message, style: const TextStyle(color: Colors.redAccent, fontSize: 13))),
      ]),
    );
  }
}

// Fun OOD (Object Not Recognized) Pop-up Widget
class _OodPopup extends StatefulWidget {
  final VoidCallback onRetry, onSelect;
  const _OodPopup({required this.onRetry, required this.onSelect});

  @override
  State<_OodPopup> createState() => _OodPopupState();
}

class _OodPopupState extends State<_OodPopup> with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _scaleAnimation;
  late Animation<double> _fadeAnimation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 600),
      vsync: this,
    );
    _scaleAnimation = Tween<double>(begin: 0.5, end: 1.0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.elasticOut),
    );
    _fadeAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeIn),
    );
    _controller.forward();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        return Opacity(
          opacity: _fadeAnimation.value,
          child: Transform.scale(
            scale: _scaleAnimation.value,
            child: child,
          ),
        );
      },
      child: Container(
        padding: const EdgeInsets.all(24),
        decoration: BoxDecoration(
          gradient: const LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [Color(0xFF4A148C), Color(0xFF7B1FA2)],
          ),
          borderRadius: BorderRadius.circular(24),
          boxShadow: [
            BoxShadow(
              color: Colors.purple.withValues(alpha: 0.4),
              blurRadius: 20,
              offset: const Offset(0, 10),
            ),
          ],
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // Emoji header
            const Text(
              '🍌🍎🍊',
              style: TextStyle(fontSize: 48),
            ),
            const SizedBox(height: 16),
            // Fun title
            const Text(
              'Oops! 🤔',
              style: TextStyle(
                color: Colors.white,
                fontSize: 28,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            // Message
            const Text(
              "Couldn't recognise this",
              style: TextStyle(
                color: Colors.white,
                fontSize: 20,
                fontWeight: FontWeight.w600,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            const Text(
              "Draw a box around the fruit or vegetable,\nor try a closer photo of it.\n"
              "Supported: apple, banana, bitter gourd,\ncapsicum, orange, tomato.",
              style: TextStyle(
                color: Colors.white70,
                fontSize: 14,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 20),
            ElevatedButton.icon(
              onPressed: widget.onSelect,
              icon: const Icon(Icons.crop_free, color: Color(0xFF4A148C)),
              label: const Text(
                'Select the fruit',
                style: TextStyle(color: Color(0xFF4A148C), fontWeight: FontWeight.bold),
              ),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(30)),
              ),
            ),
            const SizedBox(height: 10),
            // Retry button
            ElevatedButton.icon(
              onPressed: widget.onRetry,
              icon: const Icon(Icons.camera_alt, color: Colors.white),
              label: const Text(
                'Try Again 📸',
                style: TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                ),
              ),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.purple.shade700,
                padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(30),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
