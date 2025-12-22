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
  bool _isGenerating = false; 

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
      // 1. FIX: Lowered resolution from 'high' to 'medium' to prevent memory crashes
      ResolutionPreset.medium, 
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
          // Camera Preview
          Center(child: CameraPreview(_controller!)),
          // Overlay Header
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

          // Bottom Controls
          Positioned(
            bottom: 30,
            left: 0,
            right: 0,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                // Gallery / Demo Button
                GestureDetector(
                  onLongPress: () async {
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(content: Text('⚡ Entering Demo Mode...')),
                    );
                    
                    // 2. FIX: Pause camera before navigating to save resources
                    await _controller?.pausePreview(); 
                    
                    final mockData = _recipeService.getMockRecipe();
                    
                    if (!mounted) return;
                    await Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => RecipeScreen(recipeData: mockData),
                      ),
                    );

                    // Resume when coming back
                    _controller?.resumePreview(); 
                  },
                  child: IconButton(
                    icon: const Icon(Icons.photo_library, color: Colors.white),
                    onPressed: () {},
                  ),
                ),

                // Shutter Button
                FloatingActionButton.large(
                  backgroundColor: Colors.white,
                  child: const Icon(Icons.camera_alt, color: Colors.teal),
                  onPressed: () async {
                    if (!_isCameraInitialized) return;

                    setState(() => _isGenerating = true); 

                    try {
                      final image = await _controller!.takePicture();
                      
                      final ingredients = await _labelingService.processImage(image.path);
                      
                      final specificIngredients = ingredients.where((i) => i != "Food" && i != "Vegetable").toList();

                      if (specificIngredients.isEmpty) {
                        if(mounted) {
                           ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(content: Text('Too generic! Try getting closer.')),
                          );
                        }
                        setState(() => _isGenerating = false);
                        return;
                      }

                      print("Sending to Gemini: $specificIngredients");

                      final recipeJson = await _recipeService.generateRecipe(specificIngredients);

                      if (recipeJson != null && mounted) {
                        // 2. FIX: Pause camera before navigating
                        await _controller?.pausePreview();

                        await Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (context) => RecipeScreen(recipeData: recipeJson),
                          ),
                        );
                        
                        // Resume when coming back
                        _controller?.resumePreview();
                      } 
                    } catch (e) {
                      print("Error: $e");
                    } finally {
                      if (mounted) {
                        setState(() => _isGenerating = false); 
                      }
                    }
                  },
                ),

                // Settings Placeholder
                IconButton(
                  icon: const Icon(Icons.settings, color: Colors.white),
                  onPressed: () {},
                ),
              ],
            ),
          ),

          // Loading Overlay
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
                      "Chef Gemini is thinking...",
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