import 'package:flutter/material.dart';

void main() {
  runApp(const FoodyCameraApp());
}

class FoodyCameraApp extends StatelessWidget {
  const FoodyCameraApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        primaryColor: Colors.teal,
      ),
      home: const CameraCaptureScreen(),
    );
  }
}

class CameraCaptureScreen extends StatelessWidget {
  const CameraCaptureScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          // 1. The "Camera Feed" (Background Image)
          // We use a high-quality image of ingredients to simulate what the camera sees
          Image.network(
            'https://images.unsplash.com/photo-1546069901-ba9599a7e63c?ixlib=rb-1.2.1&auto=format&fit=crop&w=1000&q=80',
            fit: BoxFit.cover,
          ),

          // 2. The Camera Overlay UI
          SafeArea(
            child: Column(
              children: [
                // Top Bar (Flash, Settings)
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 20),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      _buildGlassIcon(Icons.flash_off),
                      const Text(
                        "Foody AI Lens",
                        style: TextStyle(
                          color: Colors.white, 
                          fontWeight: FontWeight.bold, 
                          fontSize: 18,
                          shadows: [Shadow(blurRadius: 10, color: Colors.black, offset: Offset(0, 2))]
                        ),
                      ),
                      _buildGlassIcon(Icons.settings),
                    ],
                  ),
                ),
                
                const Spacer(),

                // Center: AI Focus Box (The "Scanning" effect)
                Container(
                  width: 300,
                  height: 300,
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.white.withOpacity(0.5), width: 1),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          _cornerBox(true, true), // Top Left
                          _cornerBox(true, false), // Top Right
                        ],
                      ),
                      // Animated scanning text effect simulation
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                        decoration: BoxDecoration(
                          color: Colors.teal.withOpacity(0.8),
                          borderRadius: BorderRadius.circular(20),
                        ),
                        child: const Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(Icons.auto_awesome, color: Colors.white, size: 16),
                            SizedBox(width: 8),
                            Text("Detecting Ingredients...", style: TextStyle(color: Colors.white)),
                          ],
                        ),
                      ),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          _cornerBox(false, true), // Bottom Left
                          _cornerBox(false, false), // Bottom Right
                        ],
                      ),
                    ],
                  ),
                ),

                const Spacer(),

                // Bottom Control Bar
                Container(
                  padding: const EdgeInsets.only(bottom: 30, top: 20),
                  width: double.infinity,
                  color: Colors.black.withOpacity(0.3),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      // Gallery Icon
                      _buildGlassIcon(Icons.photo_library_outlined),

                      // Shutter Button
                      Container(
                        width: 80,
                        height: 80,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          border: Border.all(color: Colors.white, width: 4),
                          color: Colors.transparent,
                        ),
                        child: Padding(
                          padding: const EdgeInsets.all(4.0),
                          child: Container(
                            decoration: const BoxDecoration(
                              shape: BoxShape.circle,
                              color: Colors.white,
                            ),
                          ),
                        ),
                      ),

                      // Flip Camera Icon
                      _buildGlassIcon(Icons.flip_camera_ios_outlined),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // Helper widget for corner brackets of the focus box
  Widget _cornerBox(bool isTop, bool isLeft) {
    return Container(
      width: 30,
      height: 30,
      decoration: BoxDecoration(
        border: Border(
          top: isTop ? const BorderSide(color: Colors.teal, width: 4) : BorderSide.none,
          bottom: !isTop ? const BorderSide(color: Colors.teal, width: 4) : BorderSide.none,
          left: isLeft ? const BorderSide(color: Colors.teal, width: 4) : BorderSide.none,
          right: !isLeft ? const BorderSide(color: Colors.teal, width: 4) : BorderSide.none,
        ),
      ),
    );
  }

  // Helper widget for translucent icons
  Widget _buildGlassIcon(IconData icon) {
    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.4),
        shape: BoxShape.circle,
      ),
      child: Icon(icon, color: Colors.white, size: 28),
    );
  }
}