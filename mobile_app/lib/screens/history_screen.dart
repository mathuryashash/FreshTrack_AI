import 'dart:io';
import 'package:flutter/material.dart';
import '../models/prediction_result.dart';
import '../services/database_service.dart';
import '../widgets/freshness_badge.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> with AutomaticKeepAliveClientMixin {
  @override
  bool get wantKeepAlive => true;

  late Future<List<PredictionResult>> _future;

  @override
  void initState() {
    super.initState();
    _load();
  }

  void _load() => setState(() { _future = DatabaseService.getRecent(); });

  Future<void> _clearAll() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        backgroundColor: const Color(0xFF1C2333),
        title: const Text('Clear history?', style: TextStyle(color: Colors.white)),
        content: const Text('This will delete all local scan history.', style: TextStyle(color: Colors.white60)),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context, false), child: const Text('Cancel')),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text('Clear', style: TextStyle(color: Colors.redAccent)),
          ),
        ],
      ),
    );
    if (confirmed == true) {
      await DatabaseService.clear();
      _load();
    }
  }

  @override
  Widget build(BuildContext context) {
    super.build(context);
    return Scaffold(
      appBar: AppBar(
        title: const Text('Scan History'),
        actions: [
          IconButton(icon: const Icon(Icons.delete_outline), onPressed: _clearAll),
        ],
      ),
      body: FutureBuilder<List<PredictionResult>>(
        future: _future,
        builder: (context, snap) {
          if (snap.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator(color: Color(0xFF00E676)));
          }
          final items = snap.data ?? [];
          if (items.isEmpty) return _EmptyHistory();
          return RefreshIndicator(
            color: const Color(0xFF00E676),
            onRefresh: () async => _load(),
            child: ListView.separated(
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
              itemCount: items.length,
              separatorBuilder: (_, __) => const SizedBox(height: 10),
              itemBuilder: (_, i) => _HistoryTile(result: items[i]),
            ),
          );
        },
      ),
    );
  }
}

class _HistoryTile extends StatelessWidget {
  final PredictionResult result;
  const _HistoryTile({required this.result});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFF1C2333),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFF252D40)),
      ),
      child: Row(children: [
        // Thumbnail
        ClipRRect(
          borderRadius: const BorderRadius.horizontal(left: Radius.circular(15)),
          child: result.imagePath != null && File(result.imagePath!).existsSync()
              ? Image.file(File(result.imagePath!), width: 72, height: 72, fit: BoxFit.cover)
              : Container(
                  width: 72, height: 72,
                  color: const Color(0xFF252D40),
                  child: const Icon(Icons.image_not_supported_outlined, color: Colors.white24, size: 24),
                ),
        ),
        const SizedBox(width: 14),
        Expanded(
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 12),
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Row(children: [
                FreshnessBadge(freshness: result.freshness),
                const SizedBox(width: 8),
                Text(result.quality, style: const TextStyle(color: Colors.white54, fontSize: 12)),
              ]),
              const SizedBox(height: 6),
              Text(
                '${result.shelfLifeDays.toStringAsFixed(1)} days remaining',
                style: const TextStyle(color: Colors.white, fontSize: 14, fontWeight: FontWeight.w500),
              ),
              const SizedBox(height: 4),
              Text(result.formattedDate, style: TextStyle(color: Colors.white.withOpacity(0.3), fontSize: 11)),
            ]),
          ),
        ),
        Padding(
          padding: const EdgeInsets.only(right: 16),
          child: Text(
            '${(result.freshnessConfidence * 100).toStringAsFixed(0)}%',
            style: TextStyle(
              color: result.isSafe ? const Color(0xFF00E676) : const Color(0xFFFF5252),
              fontSize: 13,
              fontWeight: FontWeight.w700,
            ),
          ),
        ),
      ]),
    );
  }
}

class _EmptyHistory extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
        Icon(Icons.history, size: 56, color: Colors.white.withOpacity(0.1)),
        const SizedBox(height: 16),
        Text('No scans yet', style: TextStyle(color: Colors.white.withOpacity(0.3), fontSize: 16)),
        const SizedBox(height: 6),
        Text('Your scan history will appear here', style: TextStyle(color: Colors.white.withOpacity(0.2), fontSize: 13)),
      ]),
    );
  }
}
