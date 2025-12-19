import 'package:google_mlkit_image_labeling/google_mlkit_image_labeling.dart';

class ImageLabelingService {
  late ImageLabeler _imageLabeler;

  ImageLabelingService() {
    // Configure the labeler. 
    // We set the confidence threshold to 70% to avoid weak guesses.
    final ImageLabelerOptions options = ImageLabelerOptions(confidenceThreshold: 0.2);
    _imageLabeler = ImageLabeler(options: options);
  }

Future<List<String>> processImage(String imagePath) async {
  final inputImage = InputImage.fromFilePath(imagePath);
  
  // 1. Less aggressive Blocklist (We allow "Fruit" or "Vegetable" as a backup now)
  const List<String> blockList = [
    'Table', 'Wood', 'Plate', 'Cutlery', 'Dishware', 'Tableware',
    'Serveware', 'Recipe', 'Dish', 'Cuisine', 'Ingredient'
  ];

  try {
    final List<ImageLabel> labels = await _imageLabeler.processImage(inputImage);

    // 🔴 DEBUG PRINT: See what the camera actually sees!
    print("RAW AI LABELS: ${labels.map((l) => "${l.label} (${(l.confidence * 100).toStringAsFixed(0)}%)").toList()}");

    List<String> validIngredients = [];

    for (ImageLabel label in labels) {
      // 2. Filter: Only block clearly non-food items (like Table/Plate)
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