import 'package:flutter/material.dart';
import 'camera_screen.dart';

class HomeScreen extends StatelessWidget{
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Foody'), centerTitle: true,),
      body: Center(
        child: ElevatedButton.icon(onPressed: () {
  Navigator.push(
    context,
    MaterialPageRoute(builder: (context) => const CameraScreen()),
  );
}, icon: Icon(Icons.camera_alt),label: const Text('Scan Ingredients'))
      ),
    );
  }
}