import 'package:flutter/material.dart';
import '../services/api_service.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final _urlCtrl = TextEditingController();
  final _keyCtrl = TextEditingController();
  bool _saved = false;
  bool _testing = false;
  bool? _healthy;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    _urlCtrl.text = await ApiService.getBaseUrl();
    _keyCtrl.text = (await ApiService.getApiKey()) ?? '';
  }

  Future<void> _save() async {
    await ApiService.setBaseUrl(_urlCtrl.text.trim());
    await ApiService.setApiKey(_keyCtrl.text.trim());
    setState(() { _saved = true; _healthy = null; });
    Future.delayed(const Duration(seconds: 2), () {
      if (mounted) setState(() => _saved = false);
    });
  }

  Future<void> _test() async {
    setState(() { _testing = true; _healthy = null; });
    final ok = await ApiService().checkHealth();
    if (mounted) setState(() { _testing = false; _healthy = ok; });
  }

  @override
  void dispose() {
    _urlCtrl.dispose();
    _keyCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          _SectionLabel('API Configuration'),
          const SizedBox(height: 12),
          _Field(
            controller: _urlCtrl,
            label: 'API Base URL',
            hint: 'https://your-api.fly.dev',
            icon: Icons.link,
          ),
          const SizedBox(height: 12),
          _Field(
            controller: _keyCtrl,
            label: 'API Key (optional)',
            hint: 'Leave blank if auth is disabled',
            icon: Icons.key_outlined,
            obscure: true,
          ),
          const SizedBox(height: 20),
          Row(children: [
            Expanded(
              child: _ActionButton(
                label: _saved ? 'Saved ✓' : 'Save',
                color: _saved ? const Color(0xFF00E676) : const Color(0xFF1C2333),
                textColor: _saved ? const Color(0xFF001A0D) : Colors.white,
                onTap: _save,
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: _ActionButton(
                label: _testing ? 'Testing...' : 'Test Connection',
                color: const Color(0xFF1C2333),
                textColor: Colors.white,
                onTap: _testing ? null : _test,
              ),
            ),
          ]),
          if (_healthy != null) ...[
            const SizedBox(height: 16),
            _StatusBanner(healthy: _healthy!),
          ],
          const SizedBox(height: 32),
          _SectionLabel('Free Hosting Options'),
          const SizedBox(height: 12),
          ..._hostingOptions.map((h) => _HostingTile(info: h)),
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
    return Text(text, style: TextStyle(color: Colors.white.withOpacity(0.4), fontSize: 11, fontWeight: FontWeight.w600, letterSpacing: 1.2));
  }
}

class _Field extends StatelessWidget {
  final TextEditingController controller;
  final String label, hint;
  final IconData icon;
  final bool obscure;
  const _Field({required this.controller, required this.label, required this.hint, required this.icon, this.obscure = false});

  @override
  Widget build(BuildContext context) {
    return TextField(
      controller: controller,
      obscureText: obscure,
      style: const TextStyle(color: Colors.white, fontSize: 14),
      decoration: InputDecoration(
        labelText: label,
        hintText: hint,
        prefixIcon: Icon(icon, size: 18, color: Colors.white38),
        labelStyle: const TextStyle(color: Colors.white38, fontSize: 13),
        hintStyle: const TextStyle(color: Colors.white24, fontSize: 13),
        filled: true,
        fillColor: const Color(0xFF1C2333),
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(14), borderSide: const BorderSide(color: Color(0xFF252D40))),
        enabledBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(14), borderSide: const BorderSide(color: Color(0xFF252D40))),
        focusedBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(14), borderSide: const BorderSide(color: Color(0xFF00E676))),
      ),
    );
  }
}

class _ActionButton extends StatelessWidget {
  final String label;
  final Color color, textColor;
  final VoidCallback? onTap;
  const _ActionButton({required this.label, required this.color, required this.textColor, this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        height: 48,
        decoration: BoxDecoration(
          color: color,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: const Color(0xFF252D40)),
        ),
        alignment: Alignment.center,
        child: Text(label, style: TextStyle(color: textColor, fontWeight: FontWeight.w600, fontSize: 14)),
      ),
    );
  }
}

class _StatusBanner extends StatelessWidget {
  final bool healthy;
  const _StatusBanner({required this.healthy});

  @override
  Widget build(BuildContext context) {
    final color = healthy ? const Color(0xFF00E676) : const Color(0xFFFF5252);
    final msg = healthy ? 'Connected — API is healthy' : 'Could not reach the API. Check the URL.';
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: color.withOpacity(0.08),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Row(children: [
        Icon(healthy ? Icons.check_circle_outline : Icons.error_outline, color: color, size: 18),
        const SizedBox(width: 10),
        Text(msg, style: TextStyle(color: color, fontSize: 13)),
      ]),
    );
  }
}

const _hostingOptions = [
  _HostingInfo('Fly.io', 'No cold starts, global edge, 3 free VMs', 'fly.io'),
  _HostingInfo('Railway', 'Easiest deploy, \$5 free credit/month', 'railway.app'),
  _HostingInfo('HuggingFace Spaces', 'Free CPU, great for ML APIs', 'huggingface.co/spaces'),
  _HostingInfo('Koyeb', '1 free nano instance, no sleep', 'koyeb.com'),
];

class _HostingInfo {
  final String name, description, url;
  const _HostingInfo(this.name, this.description, this.url);
}

class _HostingTile extends StatelessWidget {
  final _HostingInfo info;
  const _HostingTile({required this.info});

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
        const Icon(Icons.cloud_outlined, color: Color(0xFF00E676), size: 18),
        const SizedBox(width: 12),
        Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text(info.name, style: const TextStyle(color: Colors.white, fontSize: 14, fontWeight: FontWeight.w600)),
          const SizedBox(height: 2),
          Text(info.description, style: TextStyle(color: Colors.white.withOpacity(0.4), fontSize: 12)),
        ])),
        Text(info.url, style: TextStyle(color: Colors.white.withOpacity(0.25), fontSize: 10)),
      ]),
    );
  }
}
