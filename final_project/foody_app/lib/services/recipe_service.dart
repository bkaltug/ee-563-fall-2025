import 'dart:convert';
import 'package:google_generative_ai/google_generative_ai.dart';

class RecipeService {
  late final GenerativeModel _model;

  // ⚠️ REPLACE THIS WITH YOUR ACTUAL API KEY
  static const String _apiKey = 'AIzaSyDOZfOxhJurqNsUUFiVG9cG2RQnEgtxRdI';

  RecipeService() {
    _model = GenerativeModel(
      model: 'gemini-flash-latest', // Flash is faster and cheaper for this task
      apiKey: _apiKey,
    );
  }

  Future<Map<String, dynamic>?> generateRecipe(List<String> ingredients) async {
    final ingredientsString = ingredients.join(", ");
    
    // The Prompt: We ask for specific JSON fields to match your UI
final prompt = '''
  I have identified these items from a picture: $ingredientsString.
  
  TASK:
  1. Identify the specific edible ingredients from this list.
  2. Ignore any non-food items (like "Table", "Wood", "Plate").
  3. If the list is vague (like only "Vegetable" or "Food"), create a simple, generic recipe (like "Vegetable Stir Fry" or "Omelet") but DO NOT invent ingredients I didn't scan (like Steak or Salmon).
  
  Return ONLY a valid JSON object with this structure:
  {
    "title": "Recipe Name",
    "description": "Short description.",
    "cooking_time": "15 Min",
    "calories": "200 Kcal",
    "servings": "2 Servs",
    "ingredients_used": ["List", "of", "ingredients"],
    "instructions": ["Step 1...", "Step 2..."]
  }
''';

    try {
      final content = [Content.text(prompt)];
      final response = await _model.generateContent(content);
      
      print("Gemini Response: ${response.text}"); // Debugging

      // Clean the response (sometimes LLMs add ```json ... ``` wrappers)
      String cleanJson = response.text!.replaceAll('```json', '').replaceAll('```', '').trim();
      
      return jsonDecode(cleanJson);
    } catch (e) {
      print("Error generating recipe: $e");
      return null;
    }
  }
}