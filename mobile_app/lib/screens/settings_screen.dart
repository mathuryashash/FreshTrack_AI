import 'package:flutter/material.dart';
import '../services/classifier.dart';
import '../services/database_service.dart';

/// About + Clear history. There is nothing to configure: the model runs on
/// the phone and the app never goes online.
class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  late final Future<void> _model = Classifier.instance.load();

  Future<void> _clearHistory() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        backgroundColor: const Color(0xFF1C2333),
        title: const Text('Clear history?', style: TextStyle(color: Colors.white)),
        content: const Text('This deletes all saved scans and their photos from this phone.',
            style: TextStyle(color: Colors.white60)),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context, false), child: const Text('Cancel')),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text('Clear', style: TextStyle(color: Colors.redAccent)),
          ),
        ],
      ),
    );
    if (confirmed != true) return;
    await DatabaseService.clear();
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('History cleared')));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('About')),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          const _SectionLabel('MODEL'),
          const SizedBox(height: 12),
          FutureBuilder<void>(
            future: _model,
            builder: (context, snap) {
              final meta = Classifier.instance.meta;
              final String version;
              if (snap.hasError) {
                version = 'Failed to load';
              } else if (meta == null) {
                version = 'Loading...';
              } else {
                version = meta.version;
              }
              return _Tile(
                icon: Icons.memory,
                title: version,
                subtitle: meta == null
                    ? 'On-device ONNX Runtime'
                    : 'On-device ONNX Runtime, fp32, ${meta.imageSize}x${meta.imageSize} input'
                        '${Classifier.instance.loadMs == null ? '' : ', loaded in ${Classifier.instance.loadMs!.round()} ms'}',
              );
            },
          ),
          const _Tile(
            icon: Icons.wifi_off,
            title: 'Works offline',
            subtitle: 'Photos are analysed on this phone and never uploaded.',
          ),
          const _Tile(
            icon: Icons.info_outline,
            title: 'Estimates, not a safety test',
            subtitle: 'Quality and shelf life are derived from the freshness score. '
                'Check smell, texture and mould before eating.',
          ),
          const SizedBox(height: 24),
          const _SectionLabel('DATA'),
          const SizedBox(height: 12),
          GestureDetector(
            onTap: _clearHistory,
            child: Container(
              height: 48,
              decoration: BoxDecoration(
                color: const Color(0xFF1C2333),
                borderRadius: BorderRadius.circular(14),
                border: Border.all(color: Colors.redAccent.withValues(alpha: 0.4)),
              ),
              alignment: Alignment.center,
              child: const Text('Clear history',
                  style: TextStyle(color: Colors.redAccent, fontWeight: FontWeight.w600, fontSize: 14)),
            ),
          ),
        ],
      ),
    );
  }
}

class _SectionLabel extends StatelessWidget {
  final String text;
  const _SectionLabel(this.text);

  @override
  Widget build(BuildContext context) {
    return Text(text,
        style: TextStyle(
            color: Colors.white.withValues(alpha: 0.4), fontSize: 11, fontWeight: FontWeight.w600, letterSpacing: 1.2));
  }
}

class _Tile extends StatelessWidget {
  final IconData icon;
  final String title, subtitle;
  const _Tile({required this.icon, required this.title, required this.subtitle});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFF1C2333),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: const Color(0xFF252D40)),
      ),
      child: Row(children: [
        Icon(icon, color: const Color(0xFF00E676), size: 18),
        const SizedBox(width: 12),
        Expanded(
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text(title, style: const TextStyle(color: Colors.white, fontSize: 14, fontWeight: FontWeight.w600)),
            const SizedBox(height: 2),
            Text(subtitle, style: TextStyle(color: Colors.white.withValues(alpha: 0.4), fontSize: 12)),
          ]),
        ),
      ]),
    );
  }
}
