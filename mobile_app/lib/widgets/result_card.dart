import 'package:flutter/material.dart';
import 'package:percent_indicator/circular_percent_indicator.dart';
import '../models/prediction_result.dart';

class ResultCard extends StatelessWidget {
  final PredictionResult result;
  const ResultCard({super.key, required this.result});

  Color get _statusColor {
    switch (result.freshness) {
      case 'Fresh':     return const Color(0xFF00E676);
      case 'Semi-ripe': return const Color(0xFFB2FF59);
      case 'Overripe':  return const Color(0xFFFFD740);
      default:          return const Color(0xFFFF5252);
    }
  }

  IconData get _statusIcon {
    switch (result.freshness) {
      case 'Fresh':     return Icons.check_circle_outline;
      case 'Semi-ripe': return Icons.info_outline;
      case 'Overripe':  return Icons.warning_amber_outlined;
      default:          return Icons.cancel_outlined;
    }
  }

  @override
  Widget build(BuildContext context) {
    final color = _statusColor;

    return Container(
      decoration: BoxDecoration(
        color: const Color(0xFF1C2333),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: color.withOpacity(0.25)),
      ),
      child: Column(
        children: [
          // Header
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
            decoration: BoxDecoration(
              color: color.withOpacity(0.07),
              borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
            ),
            child: Row(children: [
              Icon(_statusIcon, color: color, size: 22),
              const SizedBox(width: 10),
              Text(
                result.freshness,
                style: TextStyle(color: color, fontSize: 18, fontWeight: FontWeight.w700),
              ),
              const Spacer(),
              _SafetyBadge(isSafe: result.isSafe),
            ]),
          ),

          Padding(
            padding: const EdgeInsets.all(20),
            child: Row(
              children: [
                // Confidence ring
                CircularPercentIndicator(
                  radius: 44,
                  lineWidth: 7,
                  percent: result.freshnessConfidence.clamp(0.0, 1.0),
                  center: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
                    Text(
                      '${(result.freshnessConfidence * 100).toStringAsFixed(0)}%',
                      style: const TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.w700),
                    ),
                    Text('conf', style: TextStyle(color: Colors.white.withOpacity(0.4), fontSize: 10)),
                  ]),
                  progressColor: color,
                  backgroundColor: const Color(0xFF252D40),
                  circularStrokeCap: CircularStrokeCap.round,
                  animation: true,
                  animationDuration: 600,
                ),
                const SizedBox(width: 20),
                Expanded(
                  child: Column(children: [
                    _MetricRow(label: 'Quality', value: result.quality, color: Colors.white),
                    const SizedBox(height: 14),
                    _MetricRow(
                      label: 'Shelf Life',
                      value: '${result.shelfLifeDays.toStringAsFixed(1)} days',
                      color: _shelfLifeColor(result.shelfLifeDays),
                    ),
                    const SizedBox(height: 14),
                    _MetricRow(label: 'Scanned', value: result.formattedDate, color: Colors.white54),
                  ]),
                ),
              ],
            ),
          ),

          // Shelf life bar
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
            child: _ShelfLifeBar(days: result.shelfLifeDays, maxDays: 14),
          ),
        ],
      ),
    );
  }

  Color _shelfLifeColor(double days) {
    if (days >= 5) return const Color(0xFF00E676);
    if (days >= 2) return const Color(0xFFFFD740);
    return const Color(0xFFFF5252);
  }
}

class _SafetyBadge extends StatelessWidget {
  final bool isSafe;
  const _SafetyBadge({required this.isSafe});

  @override
  Widget build(BuildContext context) {
    final color = isSafe ? const Color(0xFF00E676) : const Color(0xFFFF5252);
    final label = isSafe ? 'Safe' : 'Avoid';
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: color.withOpacity(0.12),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Text(label, style: TextStyle(color: color, fontSize: 12, fontWeight: FontWeight.w600)),
    );
  }
}

class _MetricRow extends StatelessWidget {
  final String label, value;
  final Color color;
  const _MetricRow({required this.label, required this.value, required this.color});

  @override
  Widget build(BuildContext context) {
    return Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
      Text(label, style: TextStyle(color: Colors.white.withOpacity(0.45), fontSize: 13)),
      Text(value, style: TextStyle(color: color, fontSize: 14, fontWeight: FontWeight.w600)),
    ]);
  }
}

class _ShelfLifeBar extends StatelessWidget {
  final double days, maxDays;
  const _ShelfLifeBar({required this.days, required this.maxDays});

  @override
  Widget build(BuildContext context) {
    final pct = (days / maxDays).clamp(0.0, 1.0);
    final color = pct > 0.4 ? const Color(0xFF00E676) : pct > 0.15 ? const Color(0xFFFFD740) : const Color(0xFFFF5252);

    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
        Text('Shelf life remaining', style: TextStyle(color: Colors.white.withOpacity(0.4), fontSize: 11)),
        Text('${days.toStringAsFixed(1)} / ${maxDays.toInt()} days', style: TextStyle(color: Colors.white.withOpacity(0.4), fontSize: 11)),
      ]),
      const SizedBox(height: 6),
      ClipRRect(
        borderRadius: BorderRadius.circular(4),
        child: TweenAnimationBuilder<double>(
          tween: Tween(begin: 0, end: pct),
          duration: const Duration(milliseconds: 700),
          curve: Curves.easeOut,
          builder: (_, value, __) => LinearProgressIndicator(
            value: value,
            minHeight: 6,
            backgroundColor: const Color(0xFF252D40),
            valueColor: AlwaysStoppedAnimation(color),
          ),
        ),
      ),
    ]);
  }
}
