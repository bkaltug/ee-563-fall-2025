import 'dart:io';
import 'package:google_mlkit_image_labeling/google_mlkit_image_labeling.dart';

class ImageLabelingService {
  late final ImageLabeler _imageLabeler;
  
  ImageLabelingService() {
    final options = ImageLabelerOptions(confidenceThreshold: 0.5);
    _imageLabeler = ImageLabeler(options: options);
  }
  
  /// Processes an image and returns a list of detected labels (ingredients)
  Future<List<String>> processImage(File imageFile) async {
    try {
      final inputImage = InputImage.fromFile(imageFile);
      final labels = await _imageLabeler.processImage(inputImage);
      
      // Filter and extract relevant food-related labels
      final foodLabels = labels
          .where((label) => _isFoodRelated(label.label))
          .map((label) => label.label)
          .toList();
      
      // If no food labels found, return all labels
      if (foodLabels.isEmpty) {
        return labels.map((label) => label.label).toList();
      }
      
      return foodLabels;
    } catch (e) {
      throw Exception('Error processing image: ${e.toString()}');
    }
  }
  
  /// Checks if a label is likely food-related
  bool _isFoodRelated(String label) {
    final foodKeywords = [
      'food', 'fruit', 'vegetable', 'meat', 'fish', 'egg', 'dairy',
      'bread', 'cheese', 'milk', 'chicken', 'beef', 'pork', 'tomato',
      'potato', 'onion', 'carrot', 'apple', 'banana', 'orange', 'lettuce',
      'pepper', 'rice', 'pasta', 'noodle', 'sauce', 'butter', 'oil',
      'ingredient', 'produce', 'grocery', 'fresh', 'raw', 'cooked',
      'baked', 'fried', 'grilled', 'roasted', 'seafood', 'shrimp',
      'salmon', 'tuna', 'mushroom', 'spinach', 'broccoli', 'corn',
      'bean', 'pea', 'garlic', 'ginger', 'herb', 'spice', 'salt',
      'sugar', 'flour', 'cream', 'yogurt', 'lemon', 'lime', 'avocado',
      'cucumber', 'celery', 'cabbage', 'cauliflower', 'zucchini',
    ];
    
    final lowerLabel = label.toLowerCase();
    return foodKeywords.any((keyword) => lowerLabel.contains(keyword));
  }
  
  /// Disposes of resources
  void dispose() {
    _imageLabeler.close();
  }
}
