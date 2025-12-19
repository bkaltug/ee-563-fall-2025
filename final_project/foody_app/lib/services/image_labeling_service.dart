import 'package:google_mlkit_image_labeling/google_mlkit_image_labeling.dart';

class ImageLabelingService {
  late ImageLabeler _imageLabeler;

  ImageLabelingService() {
    // Low threshold (0.2) to catch hard-to-see items like Eggs
    final ImageLabelerOptions options = ImageLabelerOptions(confidenceThreshold: 0.2);
    _imageLabeler = ImageLabeler(options: options);
  }

  Future<List<String>> processImage(String imagePath) async {
    final inputImage = InputImage.fromFilePath(imagePath);
    
    // Blocklist: Ignore these if they appear
    const List<String> blockList = [
      'Table', 'Wood', 'Plate', 'Cutlery', 'Dishware', 'Tableware',
      'Serveware', 'Recipe', 'Dish', 'Cuisine', 'Ingredient', 'Room'
    ];

    try {
      final List<ImageLabel> labels = await _imageLabeler.processImage(inputImage);

      // Debug Print: Helps you see what the AI is actually thinking
      print("RAW AI LABELS: ${labels.map((l) => "${l.label} (${(l.confidence * 100).toStringAsFixed(0)}%)").toList()}");

      List<String> validIngredients = [];

      for (ImageLabel label in labels) {
        // Filter: Only keep if NOT in blocklist
        if (!blockList.contains(label.label)) {
          validIngredients.add(label.label);
        }
      }

      return validIngredients;
    } catch (e) {
      print("Error processing image: $e");
      return [];
    }
  }

  void dispose() {
    _imageLabeler.close();
  }
}