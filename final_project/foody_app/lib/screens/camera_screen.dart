import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import '../services/image_labeling_service.dart';
import '../services/recipe_service.dart';
import 'recipe_screen.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _controller;
  bool _isCameraInitialized = false;
  final ImageLabelingService _labelingService = ImageLabelingService();
  final RecipeService _recipeService = RecipeService();

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    var status = await Permission.camera.request();
    if (status.isDenied) {
      // Handle permission denied (show a dialog or return)
      return;
    }

    // 2. Get list of available cameras
    final cameras = await availableCameras();
    if (cameras.isEmpty) return;

    // 3. Select the first camera
    _controller = CameraController(
      cameras.first,
      ResolutionPreset.high, // High resolution for better AI detection
      enableAudio: false,
    );

    // 4. Initialize the controller
    await _controller!.initialize();

    if (mounted) {
      setState(() {
        _isCameraInitialized = true;
      });
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    _labelingService.dispose();
    super.dispose();
  }

//init
  @override
  Widget build(BuildContext context) {
    if (!_isCameraInitialized) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          // A. The Camera Feed
          Center(
            child: CameraPreview(_controller!),
          ),

          // B. The Top Bar (Overlay)
          Positioned(
            top: 50,
            left: 0,
            right: 0,
            child: Center(
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                decoration: BoxDecoration(
                  color: Colors.black54,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: const Text(
                  "Foody AI Lens",
                  style: TextStyle(color: Colors.white, fontSize: 16),
                ),
              ),
            ),
          ),

          // C. The Capture Button Area
          Positioned(
            bottom: 30,
            left: 0,
            right: 0,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                // Gallery Button (Placeholder)
                IconButton(
                  icon: const Icon(Icons.photo_library, color: Colors.white),
                  onPressed: () {},
                ),
                // Shutter Button
                FloatingActionButton.large(
                  backgroundColor: Colors.white,
                  child: const Icon(Icons.camera_alt, color: Colors.teal),
                  onPressed: () async {
                    if (!_isCameraInitialized) return;

                    try {
                      // 1. Take Picture
                      final image = await _controller!.takePicture();
                      
                      // Show Loading
                      ScaffoldMessenger.of(context).showSnackBar(
                        const SnackBar(content: Text('Processing... This AI magic takes a moment! 🪄')),
                      );

                    // 2. Get Ingredients
                          final ingredients = await _labelingService.processImage(image.path);
                          // FILTERING LOGIC: Remove generic terms
                          final specificIngredients = ingredients.where((i) => i != "Food" && i != "Vegetable").toList();

                          if (specificIngredients.isEmpty) {
                            ScaffoldMessenger.of(context).showSnackBar(
                              const SnackBar(content: Text('Too generic! Try getting closer to the ingredients.')),
                            );
                            return;
                          }
                          print("Sending these to Gemini: $specificIngredients"); // Debug print
                          // 3. Get Recipe
                         final recipeJson = await _recipeService.generateRecipe(specificIngredients);
                          ScaffoldMessenger.of(context).hideCurrentSnackBar();
                          if (recipeJson != null) {
                            // NAVIGATE TO RECIPE SCREEN
                            if (!mounted) return; // Check if user is still on this screen

                            Navigator.push(
                              context,
                              MaterialPageRoute(
                                builder: (context) => RecipeScreen(recipeData: recipeJson),
                              ),
                            );
                          } else {
                            print("Failed to generate recipe.");
                          }

                    } catch (e) {
                      print("Error: $e");
                    }
                  },
                ),
                // Settings Button (Placeholder)
                IconButton(
                  icon: const Icon(Icons.settings, color: Colors.white),
                  onPressed: () {},
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}