import 'dart:convert';
import 'dart:io';

import 'package:integration_test/integration_test_driver.dart';

/// Host side of `flutter drive`: saves the test's reportData.
Future<void> main() => integrationDriver(
      responseDataCallback: (data) async {
        if (data == null) return;
        await File('build/parity_report.json')
            .writeAsString(const JsonEncoder.withIndent(' ').convert(data));
      },
    );
