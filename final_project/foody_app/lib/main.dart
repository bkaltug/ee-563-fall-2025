import 'package:flutter/material.dart';
import 'package:foody/screens/home_screen.dart';

void main() {
  runApp(const Foody());
}

class Foody extends StatelessWidget{
  const Foody({super.key});

  @override
  Widget build(BuildContext context) {
    
    return MaterialApp(
      title: 'Foody',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal),
        useMaterial3: true
      ),
      home: const HomeScreen(),
    );
  }

  
}