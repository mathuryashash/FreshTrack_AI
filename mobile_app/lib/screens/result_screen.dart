import 'package:flutter/material.dart';
import '../models/prediction_result.dart';
import '../widgets/result_card.dart';

class ResultScreen extends StatelessWidget {
  final PredictionResult result;
  const ResultScreen({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Scan Result'),
        backgroundColor: const Color(0xFF0A0E1A),
      ),
      backgroundColor: const Color(0xFF0A0E1A),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: ResultCard(result: result),
      ),
    );
  }
}
