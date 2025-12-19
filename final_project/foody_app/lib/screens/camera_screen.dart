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
  bool _isGenerating = false; // Controls the loading overlay

  final ImageLabelingService _labelingService = ImageLabelingService();
  final RecipeService _recipeService = RecipeService();

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    var status = await Permission.camera.request();
    if (status.isDenied) return;

    final cameras = await availableCameras();
    if (cameras.isEmpty) return;

    _controller = CameraController(
      cameras.first,
      ResolutionPreset.high,
      enableAudio: false,
    );

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
          Center(child: CameraPreview(_controller!)),

          // B. The Top Overlay
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

          // C. The Capture UI
          Positioned(
            bottom: 30,
            left: 0,
            right: 0,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                // 1. Gallery / DEMO BUTTON (Long Press)
                GestureDetector(
                  onLongPress: () {
                    // 🚨 EMERGENCY DEMO MODE
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(content: Text('⚡ Entering Demo Mode...')),
                    );
                    
                    final mockData = _recipeService.getMockRecipe();
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => RecipeScreen(recipeData: mockData),
                      ),
                    );
                  },
                  child: IconButton(
                    icon: const Icon(Icons.photo_library, color: Colors.white),
                    onPressed: () {
                       // Optional: Normal gallery logic
                    },
                  ),
                ),

                // 2. Shutter Button
                FloatingActionButton.large(
                  backgroundColor: Colors.white,
                  child: const Icon(Icons.camera_alt, color: Colors.teal),
                  onPressed: () async {
                    if (!_isCameraInitialized) return;

                    setState(() => _isGenerating = true); // Start Loading

                    try {
                      // 1. Take Picture
                      final image = await _controller!.takePicture();
                      
                      // 2. Get Ingredients
                      final ingredients = await _labelingService.processImage(image.path);
                      
                      // Filter generic noise
                      final specificIngredients = ingredients.where((i) => i != "Food" && i != "Vegetable").toList();

                      if (specificIngredients.isEmpty) {
                        if(mounted) {
                           ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(content: Text('Too generic! Try getting closer.')),
                          );
                        }
                        return;
                      }

                      print("Sending to Gemini: $specificIngredients");

                      // 3. Get Recipe
                      final recipeJson = await _recipeService.generateRecipe(specificIngredients);

                      if (recipeJson != null && mounted) {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (context) => RecipeScreen(recipeData: recipeJson),
                          ),
                        );
                      } 
                    } catch (e) {
                      print("Error: $e");
                    } finally {
                      if (mounted) {
                        setState(() => _isGenerating = false); // Stop Loading
                      }
                    }
                  },
                ),

                // 3. Settings Placeholder
                IconButton(
                  icon: const Icon(Icons.settings, color: Colors.white),
                  onPressed: () {},
                ),
              ],
            ),
          ),

          // D. Loading Overlay
          if (_isGenerating)
            Container(
              color: Colors.black54,
              child: Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const CircularProgressIndicator(color: Colors.teal),
                    const SizedBox(height: 20),
                    const Text(
                      "Foody is creating your recipe...",
                      style: TextStyle(color: Colors.white, fontSize: 18),
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }
}