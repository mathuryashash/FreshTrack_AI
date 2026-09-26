import 'dart:io';

import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';

import '../services/classifier.dart';
import '../services/pipeline.dart';

const freshColor = Color(0xFF00E676), staleColor = Color(0xFFFF5252), unknownColor = Color(0xFFB0BEC5);

Color itemColor(ScanItem i) => !i.classification.isProduce
    ? unknownColor
    : (i.classification.freshness == 'Fresh' ? freshColor : staleColor);

/// Recognised items are numbered 1..n in list order; unrecognised ones get null.
List<int?> itemNumbers(List<ScanItem> items) {
  var n = 0;
  return [for (final i in items) i.classification.isProduce ? ++n : null];
}

/// The photo, letterboxed to its own aspect ratio, with a box per item. Tap a
/// box to open it. In [selecting] mode a drag draws a box, reported in photo
/// pixel coordinates.
class ScanImage extends StatefulWidget {
  final File image;
  final int imageWidth, imageHeight;
  final List<ScanItem> items;
  final bool selecting;
  final double maxHeight;
  final ValueChanged<int>? onTapItem;
  final ValueChanged<Detection>? onSelected;

  const ScanImage({
    super.key,
    required this.image,
    required this.imageWidth,
    required this.imageHeight,
    required this.items,
    this.selecting = false,
    this.maxHeight = 420,
    this.onTapItem,
    this.onSelected,
  });

  /// Smallest drawn box accepted, in screen pixels.
  static const minDrag = 16.0;

  @override
  State<ScanImage> createState() => _ScanImageState();
}

class _ScanImageState extends State<ScanImage> {
  Offset? _start, _end;

  @override
  void didUpdateWidget(ScanImage old) {
    super.didUpdateWidget(old);
    if (!widget.selecting) _start = _end = null;
  }

  Offset _clamp(Offset p, Size s) => Offset(p.dx.clamp(0, s.width), p.dy.clamp(0, s.height));

  void _tap(Offset p, double scale) {
    int? best;
    var bestArea = double.infinity;
    for (var k = 0; k < widget.items.length; k++) {
      final b = widget.items[k].box;
      if (b == null) continue;
      final r = Rect.fromLTRB(b.x1 * scale, b.y1 * scale, b.x2 * scale, b.y2 * scale);
      if (r.contains(p) && r.width * r.height < bestArea) {
        best = k;
        bestArea = r.width * r.height;
      }
    }
    if (best != null) widget.onTapItem?.call(best);
  }

  void _finish(double scale) {
    final s = _start, e = _end;
    setState(() => _start = _end = null);
    if (s == null || e == null) return;
    final r = Rect.fromPoints(s, e);
    if (r.width < ScanImage.minDrag || r.height < ScanImage.minDrag) return;
    widget.onSelected?.call(Detection(r.left / scale, r.top / scale, r.right / scale, r.bottom / scale, 1));
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(builder: (context, c) {
      final aspect = widget.imageWidth / widget.imageHeight;
      var w = c.maxWidth, h = w / aspect;
      if (h > widget.maxHeight) {
        h = widget.maxHeight;
        w = h * aspect;
      }
      final size = Size(w, h);
      final scale = w / widget.imageWidth;
      final drag = _start != null && _end != null ? Rect.fromPoints(_start!, _end!) : null;
      final n = widget.items.where((i) => i.classification.isProduce).length;
      return Center(
        child: Semantics(
          label: widget.selecting
              ? 'Photo. Drag to draw a box around one fruit or vegetable.'
              : 'Photo with $n recognised ${n == 1 ? 'item' : 'items'} marked',
          child: SizedBox(
            width: w,
            height: h,
            child: GestureDetector(
              behavior: HitTestBehavior.opaque,
              // The box starts where the finger touched down, not where the drag slop was passed.
              dragStartBehavior: DragStartBehavior.down,
              onTapUp: widget.selecting ? null : (d) => _tap(d.localPosition, scale),
              onPanStart: widget.selecting
                  ? (d) => setState(() => _start = _end = _clamp(d.localPosition, size))
                  : null,
              onPanUpdate: widget.selecting ? (d) => setState(() => _end = _clamp(d.localPosition, size)) : null,
              onPanEnd: widget.selecting ? (_) => _finish(scale) : null,
              child: Stack(fit: StackFit.expand, children: [
                Image.file(widget.image, fit: BoxFit.fill),
                CustomPaint(painter: BoxPainter(widget.items, scale, drag)),
              ]),
            ),
          ),
        ),
      );
    });
  }
}

class BoxPainter extends CustomPainter {
  final List<ScanItem> items;
  final double scale;
  final Rect? drag;
  BoxPainter(this.items, this.scale, this.drag);

  @override
  void paint(Canvas canvas, Size size) {
    final numbers = itemNumbers(items);
    for (var k = 0; k < items.length; k++) {
      final b = items[k].box;
      if (b == null) continue;
      final color = itemColor(items[k]);
      final r = Rect.fromLTRB(b.x1 * scale, b.y1 * scale, b.x2 * scale, b.y2 * scale);
      canvas.drawRRect(
        RRect.fromRectAndRadius(r, const Radius.circular(6)),
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = numbers[k] == null ? 1.5 : 2.5
          ..color = color,
      );
      final label = numbers[k]?.toString() ?? '?';
      final tp = TextPainter(
        text: TextSpan(
          text: label,
          style: const TextStyle(color: Color(0xFF0A0E1A), fontSize: 12, fontWeight: FontWeight.w800),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      const d = 20.0;
      final c = Offset((r.left + d / 2).clamp(d / 2, size.width - d / 2), (r.top + d / 2).clamp(d / 2, size.height - d / 2));
      canvas.drawCircle(c, d / 2, Paint()..color = color);
      tp.paint(canvas, c - Offset(tp.width / 2, tp.height / 2));
    }
    final dr = drag;
    if (dr != null) {
      canvas.drawRect(dr, Paint()..color = Colors.white.withValues(alpha: 0.15));
      canvas.drawRect(
        dr,
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2
          ..color = Colors.white,
      );
    }
  }

  @override
  bool shouldRepaint(BoxPainter old) => old.items != items || old.scale != scale || old.drag != drag;
}

/// One compact row per recognised item.
class ItemTile extends StatelessWidget {
  final int number;
  final ScanItem item;
  final VoidCallback onTap;
  const ItemTile({super.key, required this.number, required this.item, required this.onTap});

  @override
  Widget build(BuildContext context) {
    final c = item.classification;
    final r = c.toResult();
    final color = itemColor(item);
    return Material(
      color: const Color(0xFF1C2333),
      borderRadius: BorderRadius.circular(16),
      child: InkWell(
        borderRadius: BorderRadius.circular(16),
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: color.withValues(alpha: 0.35)),
          ),
          child: Row(children: [
            CircleAvatar(
              radius: 14,
              backgroundColor: color,
              child: Text('$number',
                  style: const TextStyle(color: Color(0xFF0A0E1A), fontSize: 13, fontWeight: FontWeight.w800)),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text(r.produceLabel ?? 'Produce',
                    style: const TextStyle(color: Colors.white, fontSize: 15, fontWeight: FontWeight.w700)),
                const SizedBox(height: 2),
                Text(
                  '${r.statusLabel} · ${(c.freshnessConfidence * 100).round()}%${item.manual ? ' · your box' : ''}',
                  style: TextStyle(color: color, fontSize: 12.5),
                ),
              ]),
            ),
            const Icon(Icons.chevron_right, color: Colors.white38),
          ]),
        ),
      ),
    );
  }
}
