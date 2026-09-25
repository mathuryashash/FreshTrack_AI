import 'package:flutter/material.dart';

class FreshnessBadge extends StatelessWidget {
  final String freshness;
  const FreshnessBadge({super.key, required this.freshness});

  Color get _color {
    // API labels since v2: 'Fresh' | 'Stale'.
    switch (freshness) {
      case 'Fresh': return const Color(0xFF00E676);
      case 'Stale': return const Color(0xFFFF5252);
      default:      return const Color(0xFF9E9E9E);
    }
  }

  @override
  Widget build(BuildContext context) {
    final color = _color;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withValues(alpha: 0.3)),
      ),
      child: Text(
        freshness,
        style: TextStyle(color: color, fontSize: 11, fontWeight: FontWeight.w600),
      ),
    );
  }
}
