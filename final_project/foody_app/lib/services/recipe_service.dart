import 'dart:convert';
import 'package:google_generative_ai/google_generative_ai.dart';

class RecipeService {
  late final GenerativeModel _model;

  // ⚠️ REPLACE WITH YOUR ACTUAL API KEY
  static const String _apiKey = 'AIzaSyDOZfOxhJurqNsUUFiVG9cG2RQnEgtxRdI';

  RecipeService() {
    // Using 'gemini-pro' as it is stable and widely available
    _model = GenerativeModel(
      model: 'gemini-flash-latest', 
      apiKey: _apiKey,
    );
  }

  Future<Map<String, dynamic>?> generateRecipe(List<String> ingredients) async {
    final ingredientsString = ingredients.join(", ");
    
    // The Prompt: Forces Gemini to ignore generic "Food" tags and output JSON
    final prompt = '''
      I have identified these items from a picture: $ingredientsString.
      
      TASK:
      1. Identify the specific edible ingredients from this list.
      2. Ignore any non-food items (like "Table", "Wood", "Plate").
      3. If the list is vague (like only "Vegetable" or "Food"), create a simple, generic recipe (like "Vegetable Stir Fry" or "Omelet") but DO NOT invent ingredients I didn't scan (like Steak or Salmon).
      
      Return ONLY a valid JSON object with this exact structure, no markdown formatting:
      {
        "title": "Recipe Name",
        "description": "Short description.",
        "cooking_time": "15 Min",
        "calories": "200 Kcal",
        "servings": "2 Servs",
        "ingredients_used": ["List", "of", "ingredients"],
        "instructions": ["Step 1...", "Step 2...", "Step 3..."]
      }
    ''';

    try {
      final content = [Content.text(prompt)];
      final response = await _model.generateContent(content);
      
      print("Gemini Response: ${response.text}"); 

      // Clean the response (remove ```json wrappers if present)
      String cleanJson = response.text!.replaceAll('```json', '').replaceAll('```', '').trim();
      
      return jsonDecode(cleanJson);
    } catch (e) {
      print("Error generating recipe: $e");
      return null;
    }
  }

  // 🚨 EMERGENCY DEMO DATA
  // Used when you long-press the Gallery button
  Map<String, dynamic> getMockRecipe() {
    return {
      "title": "Turkish Menemen",
      "description": "A traditional Turkish dish made with eggs, tomato, and green peppers.",
      "cooking_time": "15 Min",
      "calories": "250 Kcal",
      "servings": "2 Servs",
      "ingredients_used": ["Tomato", "Egg", "Green Pepper", "Oil"],
      "instructions": [
        "Heat oil in a pan and sauté the peppers.",
        "Add the chopped tomatoes and cook until soft.",
        "Crack the eggs into the pan and scramble gently.",
        "Season with salt and pepper. Serve hot with bread."
      ]
    };
  }
}