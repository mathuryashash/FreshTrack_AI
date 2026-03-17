import 'package:flutter_test/flutter_test.dart';
import 'package:freshtrack_mobile/main.dart';

void main() {
  testWidgets('FreshTrackApp renders home screen', (WidgetTester tester) async {
    await tester.pumpWidget(const FreshTrackApp());
    await tester.pumpAndSettle();

    // Verify the app title is present
    expect(find.text('FreshTrack AI'), findsWidgets);
  });
}
