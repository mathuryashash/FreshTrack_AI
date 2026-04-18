import 'package:flutter/material.dart';

class FreshnessBadge extends StatelessWidget {
  final String freshness;
  const FreshnessBadge({super.key, required this.freshness});

  Color get _color {
    switch (freshness) {
      case 'Fresh':     return const Color(0xFF00E676);
      case 'Semi-ripe': return const Color(0xFFB2FF59);
      case 'Overripe':  return const Color(0xFFFFD740);
      default:          return const Color(0xFFFF5252);
    }
  }

  @override
  Widget build(BuildContext context) {
    final color = _color;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: color.withOpacity(0.12),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Text(
        freshness,
        style: TextStyle(color: color, fontSize: 11, fontWeight: FontWeight.w600),
      ),
    );
  }
}
