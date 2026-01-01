import 'dart:io';
import 'dart:typed_data';
import 'package:google_generative_ai/google_generative_ai.dart';
import '../config/api_keys.dart';

class GeminiService {
  late final GenerativeModel _model;
  
  GeminiService() {
    _model = GenerativeModel(
      model: 'gemini-flash-latest',
      apiKey: ApiKeys.geminiApiKey,
    );
  }
  
  /// Analyzes an image and returns a list of specific food ingredients detected
  Future<List<String>> detectIngredients(File imageFile) async {
    try {
      final Uint8List imageBytes = await imageFile.readAsBytes();
      final String mimeType = _getMimeType(imageFile.path);
      
      final prompt = '''
Analyze this image and identify all the specific food ingredients you can see.

IMPORTANT RULES:
- Only list actual food items that are clearly visible in the image
- Be SPECIFIC: say "egg" not "food", say "tomato" not "vegetable", say "chicken breast" not "meat"
- List each ingredient on a new line
- Do NOT include generic categories like "food", "fruit", "vegetable", "produce", "ingredient"
- Do NOT include non-food items
- Do NOT include packaging or containers
- If you see multiple of the same item, just list it once (e.g., "tomato" not "tomato, tomato")
- Use simple names: "egg", "tomato", "onion", "garlic", "chicken", "beef", "carrot", etc.
- If you cannot identify any food items, respond with "NO_FOOD_DETECTED"

List only the ingredient names, one per line, nothing else:
''';

      final content = [
        Content.multi([
          TextPart(prompt),
          DataPart(mimeType, imageBytes),
        ])
      ];
      
      final response = await _model.generateContent(content);
      final responseText = response.text ?? '';
      
      if (responseText.contains('NO_FOOD_DETECTED') || responseText.trim().isEmpty) {
        return [];
      }
      
      // Parse the response into a list of ingredients
      final ingredients = responseText
          .split('\n')
          .map((line) => line.trim())
          .where((line) => line.isNotEmpty)
          .where((line) => !_isGenericLabel(line))
          .map((line) => _cleanIngredientName(line))
          .where((line) => line.isNotEmpty)
          .toSet() // Remove duplicates
          .toList();
      
      return ingredients;
    } catch (e) {
      throw Exception('Error detecting ingredients: ${e.toString()}');
    }
  }
  
  /// Checks if a label is too generic
  bool _isGenericLabel(String label) {
    final genericLabels = [
      'food', 'fruit', 'vegetable', 'produce', 'ingredient', 'ingredients',
      'meat', 'protein', 'dairy', 'grocery', 'item', 'items', 'object',
      'toy', 'balloon', 'ball', 'plastic', 'container', 'package',
    ];
    final lowerLabel = label.toLowerCase();
    return genericLabels.any((generic) => lowerLabel == generic);
  }
  
  /// Cleans up ingredient name (removes bullets, numbers, etc.)
  String _cleanIngredientName(String name) {
    // Remove common prefixes like "- ", "* ", "1. ", etc.
    String cleaned = name.replaceAll(RegExp(r'^[\-\*\•\d\.]+\s*'), '').trim();
    // Remove any markdown formatting
    cleaned = cleaned.replaceAll(RegExp(r'[\*\_]+'), '').trim();
    // Capitalize first letter
    if (cleaned.isNotEmpty) {
      cleaned = cleaned[0].toUpperCase() + cleaned.substring(1).toLowerCase();
    }
    return cleaned;
  }
  
  /// Gets the MIME type based on file extension
  String _getMimeType(String path) {
    final extension = path.split('.').last.toLowerCase();
    switch (extension) {
      case 'jpg':
      case 'jpeg':
        return 'image/jpeg';
      case 'png':
        return 'image/png';
      case 'gif':
        return 'image/gif';
      case 'webp':
        return 'image/webp';
      default:
        return 'image/jpeg';
    }
  }
  
  /// Generates a recipe based on the provided ingredient labels and user preferences
  Future<String> generateRecipe(
    List<String> ingredients, {
    String eatingPreference = 'No Choice',
    String glutenPreference = 'Gluten',
    String cookingSkill = 'Average',
    int servingSize = 1,
  }) async {
    if (ingredients.isEmpty) {
      return 'No ingredients detected. Please try again with a clearer image.';
    }
    
    final ingredientList = ingredients.join(', ');
    final servingText = '$servingSize ${servingSize == 1 ? 'person' : 'people'}';
    
    // Build dietary restrictions string
    String dietaryRestrictions = '';
    if (eatingPreference == 'Vegetarian') {
      dietaryRestrictions += '- The recipe MUST be vegetarian (no meat, no fish, no poultry)\n';
    } else if (eatingPreference == 'Vegan') {
      dietaryRestrictions += '- The recipe MUST be vegan (no animal products at all - no meat, fish, eggs, dairy, honey)\n';
    }
    
    if (glutenPreference == 'Gluten-Free') {
      dietaryRestrictions += '- The recipe MUST be gluten-free (no wheat, barley, rye, or gluten-containing ingredients)\n';
    }
    
    // Build skill level instructions
    String skillInstructions = '';
    if (cookingSkill == 'Average') {
      skillInstructions = 'Keep the recipe simple and straightforward, suitable for home cooks with basic skills. Use common techniques and avoid complex methods.';
    } else if (cookingSkill == 'Chef') {
      skillInstructions = 'You can suggest more advanced and sophisticated recipes with professional techniques, complex flavor combinations, and restaurant-quality presentation.';
    }
    
    final prompt = '''
You are a helpful cooking assistant. Based on the following ingredients detected from an image, suggest a delicious and easy-to-make recipe.

Detected ingredients: $ingredientList

Always available pantry staples (you can use these freely): salt, black pepper, cooking oil, water

SERVING SIZE: $servingText (adjust all ingredient quantities accordingly)

DIETARY REQUIREMENTS (MUST FOLLOW):
$dietaryRestrictions
SKILL LEVEL:
$skillInstructions

Please provide:
1. **Recipe name** as a heading
2. **Ingredients** - Use a simple bullet list (one ingredient per line with a dash or bullet). Do NOT use tables. Quantities should be for $servingText.
3. **Instructions** - Numbered step-by-step cooking instructions
4. **Cooking time** and **Serving size** (should be $servingText)

IMPORTANT FORMATTING RULES:
- Do NOT use tables or markdown tables at all
- Use simple bullet points (- or •) for ingredient lists
- Keep lines short and mobile-friendly
- Use **bold** for section headers
- Use numbered lists for steps
- Use METRIC/EUROPEAN units only: grams (g), kilograms (kg), milliliters (ml), liters (L), centimeters (cm)
- Do NOT use cups, ounces, pounds, tablespoons, teaspoons, or other imperial units
- For small amounts, use grams or ml (e.g., "5g salt" instead of "1 tsp salt", "15ml oil" instead of "1 tbsp oil")

Make the recipe practical and suitable for home cooking. If the detected ingredients are limited, suggest a simple recipe that primarily uses those ingredients with minimal additional items.
''';

    try {
      final content = [Content.text(prompt)];
      final response = await _model.generateContent(content);
      
      return response.text ?? 'Unable to generate recipe. Please try again.';
    } catch (e) {
      return 'Error generating recipe: ${e.toString()}';
    }
  }
}
