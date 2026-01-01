import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';
import '../services/gemini_service.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ImagePicker _picker = ImagePicker();
  final GeminiService _geminiService = GeminiService();

  File? _selectedImage;
  List<String> _detectedIngredients = [];
  String _recipe = '';
  bool _isProcessing = false;
  bool _isGeneratingRecipe = false;

  // User preferences
  String _eatingPreference = 'No Choice';
  String _glutenPreference = 'Gluten';
  String _cookingSkill = 'Average';
  int _servingSize = 1;

  @override
  void dispose() {
    super.dispose();
  }

  Future<void> _requestCameraPermission() async {
    final status = await Permission.camera.request();
    if (status.isDenied) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Camera permission is required')),
        );
      }
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      if (source == ImageSource.camera) {
        await _requestCameraPermission();
      }

      final XFile? image = await _picker.pickImage(
        source: source,
        maxWidth: 1024,
        maxHeight: 1024,
        imageQuality: 85,
      );

      if (image != null) {
        setState(() {
          _selectedImage = File(image.path);
          _detectedIngredients = [];
          _recipe = '';
        });
        await _processImage();
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error picking image: ${e.toString()}')),
        );
      }
    }
  }

  Future<void> _processImage() async {
    if (_selectedImage == null) return;

    setState(() {
      _isProcessing = true;
    });

    try {
      // Use Gemini Vision to detect specific ingredients
      final ingredients = await _geminiService.detectIngredients(_selectedImage!);
      setState(() {
        _detectedIngredients = ingredients;
        _isProcessing = false;
      });
      
      if (ingredients.isEmpty && mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('No food ingredients detected. Try a clearer photo.')),
        );
      }
    } catch (e) {
      setState(() {
        _isProcessing = false;
      });
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error analyzing image: ${e.toString()}')),
        );
      }
    }
  }

  Future<void> _generateRecipe() async {
    if (_detectedIngredients.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('No ingredients detected. Please take a photo first.')),
      );
      return;
    }

    setState(() {
      _isGeneratingRecipe = true;
    });

    try {
      final recipe = await _geminiService.generateRecipe(
        _detectedIngredients,
        eatingPreference: _eatingPreference,
        glutenPreference: _glutenPreference,
        cookingSkill: _cookingSkill,
        servingSize: _servingSize,
      );
      setState(() {
        _recipe = recipe;
        _isGeneratingRecipe = false;
      });
    } catch (e) {
      setState(() {
        _isGeneratingRecipe = false;
      });
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error generating recipe: ${e.toString()}')),
        );
      }
    }
  }

  void _showImageSourceDialog() {
    showModalBottomSheet(
      context: context,
      builder: (BuildContext context) {
        return SafeArea(
          child: Wrap(
            children: [
              ListTile(
                leading: const Icon(Icons.camera_alt),
                title: const Text('Take a Photo'),
                onTap: () {
                  Navigator.pop(context);
                  _pickImage(ImageSource.camera);
                },
              ),
              ListTile(
                leading: const Icon(Icons.photo_library),
                title: const Text('Choose from Gallery'),
                onTap: () {
                  Navigator.pop(context);
                  _pickImage(ImageSource.gallery);
                },
              ),
            ],
          ),
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Foody',style: TextStyle(color: (Colors.white)),),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Image Preview Section
            Card(
              elevation: 4,
              child: Container(
                height: 250,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(12),
                  color: Colors.grey[200],
                ),
                child: _selectedImage != null
                    ? ClipRRect(
                        borderRadius: BorderRadius.circular(12),
                        child: Image.file(
                          _selectedImage!,
                          fit: BoxFit.cover,
                          width: double.infinity,
                        ),
                      )
                    : Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(
                              Icons.camera_alt,
                              size: 64,
                              color: Colors.grey[400],
                            ),
                            const SizedBox(height: 8),
                            Text(
                              'Take a photo of your ingredients',
                              style: TextStyle(
                                color: Colors.grey[600],
                                fontSize: 16,
                              ),
                            ),
                          ],
                        ),
                      ),
              ),
            ),
            const SizedBox(height: 16),

            // Capture Button
            ElevatedButton.icon(
              onPressed: _isProcessing ? null : _showImageSourceDialog,
              icon: const Icon(Icons.camera_alt),
              label: const Text('Capture Ingredients'),
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 16),
                backgroundColor: Theme.of(context).colorScheme.primary,
                foregroundColor: Colors.white,
              ),
            ),
            const SizedBox(height: 20),

            // Preferences Section
            Card(
              elevation: 2,
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(Icons.tune, color: Colors.purple.shade600),
                        const SizedBox(width: 8),
                        Text(
                          'Recipe Preferences',
                          style: Theme.of(context).textTheme.titleMedium?.copyWith(
                                fontWeight: FontWeight.bold,
                              ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 16),
                    
                    // Eating Preference
                    Text(
                      'Eating Preference',
                      style: TextStyle(
                        fontWeight: FontWeight.w600,
                        color: Colors.grey.shade700,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.symmetric(horizontal: 12),
                      decoration: BoxDecoration(
                        border: Border.all(color: Colors.grey.shade300),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: DropdownButtonHideUnderline(
                        child: DropdownButton<String>(
                          value: _eatingPreference,
                          isExpanded: true,
                          items: ['No Choice', 'Vegetarian', 'Vegan']
                              .map((e) => DropdownMenuItem(value: e, child: Text(e)))
                              .toList(),
                          onChanged: (value) {
                            setState(() => _eatingPreference = value!);
                          },
                        ),
                      ),
                    ),
                    const SizedBox(height: 16),
                    
                    // Gluten Preference
                    Text(
                      'Gluten Preference',
                      style: TextStyle(
                        fontWeight: FontWeight.w600,
                        color: Colors.grey.shade700,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.symmetric(horizontal: 12),
                      decoration: BoxDecoration(
                        border: Border.all(color: Colors.grey.shade300),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: DropdownButtonHideUnderline(
                        child: DropdownButton<String>(
                          value: _glutenPreference,
                          isExpanded: true,
                          items: ['Gluten', 'Gluten-Free']
                              .map((e) => DropdownMenuItem(value: e, child: Text(e)))
                              .toList(),
                          onChanged: (value) {
                            setState(() => _glutenPreference = value!);
                          },
                        ),
                      ),
                    ),
                    const SizedBox(height: 16),
                    
                    // Cooking Skills
                    Text(
                      'Cooking Skills',
                      style: TextStyle(
                        fontWeight: FontWeight.w600,
                        color: Colors.grey.shade700,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.symmetric(horizontal: 12),
                      decoration: BoxDecoration(
                        border: Border.all(color: Colors.grey.shade300),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: DropdownButtonHideUnderline(
                        child: DropdownButton<String>(
                          value: _cookingSkill,
                          isExpanded: true,
                          items: ['Average', 'Chef']
                              .map((e) => DropdownMenuItem(value: e, child: Text(e)))
                              .toList(),
                          onChanged: (value) {
                            setState(() => _cookingSkill = value!);
                          },
                        ),
                      ),
                    ),
                    const SizedBox(height: 16),
                    
                    // Serving Size
                    Text(
                      'Serving Size',
                      style: TextStyle(
                        fontWeight: FontWeight.w600,
                        color: Colors.grey.shade700,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.symmetric(horizontal: 12),
                      decoration: BoxDecoration(
                        border: Border.all(color: Colors.grey.shade300),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: DropdownButtonHideUnderline(
                        child: DropdownButton<int>(
                          value: _servingSize,
                          isExpanded: true,
                          items: [1, 2, 3, 4]
                              .map((e) => DropdownMenuItem(value: e, child: Text('$e ${e == 1 ? 'person' : 'people'}')))
                              .toList(),
                          onChanged: (value) {
                            setState(() => _servingSize = value!);
                          },
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),

            // Processing Indicator
            if (_isProcessing)
              const Center(
                child: Column(
                  children: [
                    CircularProgressIndicator(),
                    SizedBox(height: 8),
                    Text('Analyzing image...'),
                  ],
                ),
              ),

            // Detected Ingredients Section
            if (_detectedIngredients.isNotEmpty) ...[
              Card(
                elevation: 2,
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          const Icon(Icons.restaurant_menu, color: Colors.green),
                          const SizedBox(width: 8),
                          Text(
                            'Detected Ingredients',
                            style: Theme.of(context).textTheme.titleMedium?.copyWith(
                                  fontWeight: FontWeight.bold,
                                ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 12),
                      Wrap(
                        spacing: 8,
                        runSpacing: 8,
                        children: _detectedIngredients.map((ingredient) {
                          return Chip(
                            label: Text(ingredient),
                            backgroundColor: Colors.green[100],
                            labelStyle: const TextStyle(color: Colors.green),
                          );
                        }).toList(),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 16),

              // Generate Recipe Button
              ElevatedButton.icon(
                onPressed: _isGeneratingRecipe ? null : _generateRecipe,
                icon: _isGeneratingRecipe
                    ? const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: Colors.white,
                        ),
                      )
                    : const Icon(Icons.auto_awesome),
                label: Text(_isGeneratingRecipe ? 'Generating Recipe...' : 'Generate Recipe'),
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  backgroundColor: Colors.orange,
                  foregroundColor: Colors.white,
                ),
              ),
            ],
            const SizedBox(height: 16),

            // Recipe Section
            if (_recipe.isNotEmpty)
              Card(
                elevation: 4,
                shadowColor: Colors.orange.withOpacity(0.3),
                child: Container(
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(12),
                    gradient: LinearGradient(
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                      colors: [
                        Colors.orange.shade50,
                        Colors.white,
                      ],
                    ),
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(20.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Container(
                              padding: const EdgeInsets.all(8),
                              decoration: BoxDecoration(
                                color: Colors.orange.shade100,
                                borderRadius: BorderRadius.circular(8),
                              ),
                              child: const Icon(Icons.restaurant, color: Colors.orange, size: 24),
                            ),
                            const SizedBox(width: 12),
                            Text(
                              'Your Recipe',
                              style: Theme.of(context).textTheme.titleLarge?.copyWith(
                                    fontWeight: FontWeight.bold,
                                    color: Colors.orange.shade800,
                                  ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 16),
                        Container(
                          height: 2,
                          decoration: BoxDecoration(
                            gradient: LinearGradient(
                              colors: [Colors.orange.shade300, Colors.orange.shade100, Colors.transparent],
                            ),
                          ),
                        ),
                        const SizedBox(height: 16),
                        MarkdownBody(
                          data: _recipe,
                          selectable: true,
                          styleSheet: MarkdownStyleSheet(
                            h1: TextStyle(
                              fontSize: 24,
                              fontWeight: FontWeight.bold,
                              color: Colors.green.shade800,
                              height: 1.4,
                            ),
                            h2: TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.bold,
                              color: Colors.orange.shade700,
                              height: 1.4,
                            ),
                            h3: TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.w600,
                              color: Colors.brown.shade700,
                              height: 1.4,
                            ),
                            p: const TextStyle(
                              fontSize: 15,
                              height: 1.6,
                              color: Colors.black87,
                            ),
                            listBullet: TextStyle(
                              fontSize: 15,
                              color: Colors.orange.shade600,
                            ),
                            listIndent: 20,
                            blockSpacing: 12,
                            strong: const TextStyle(
                              fontWeight: FontWeight.bold,
                              color: Colors.black87,
                            ),
                            em: TextStyle(
                              fontStyle: FontStyle.italic,
                              color: Colors.grey.shade700,
                            ),
                            blockquoteDecoration: BoxDecoration(
                              color: Colors.orange.shade50,
                              borderRadius: BorderRadius.circular(8),
                              border: Border(
                                left: BorderSide(
                                  color: Colors.orange.shade300,
                                  width: 4,
                                ),
                              ),
                            ),
                            blockquotePadding: const EdgeInsets.all(12),
                            codeblockDecoration: BoxDecoration(
                              color: Colors.grey.shade100,
                              borderRadius: BorderRadius.circular(8),
                            ),
                            codeblockPadding: const EdgeInsets.all(12),
                            horizontalRuleDecoration: BoxDecoration(
                              border: Border(
                                top: BorderSide(
                                  color: Colors.orange.shade200,
                                  width: 1,
                                ),
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
