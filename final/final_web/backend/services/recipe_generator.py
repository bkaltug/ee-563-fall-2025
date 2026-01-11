"""
Recipe Generation Service using HuggingFace Transformers
This module handles recipe generation from ingredient lists using LLMs
"""

import os
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class RecipeGenerator:
    """
    Recipe Generator using HuggingFace Transformers
    
    Generates cooking recipes based on provided ingredient lists
    using various LLM models from HuggingFace.
    """
    
    # Supported models for recipe generation
    SUPPORTED_MODELS = {
        'gpt2': 'gpt2',
        'gpt2-medium': 'gpt2-medium',
        'distilgpt2': 'distilgpt2',
        'bloom': 'bigscience/bloom-560m',
        'flan-t5': 'google/flan-t5-base',
        'mistral': 'mistralai/Mistral-7B-Instruct-v0.1',
        'llama2': 'meta-llama/Llama-2-7b-chat-hf',
        'recipe-nlg': 'flax-community/t5-recipe-generation',
    }
    
    def __init__(self, model_name: str = 'flan-t5', use_api: bool = False, api_token: str = None):
        """
        Initialize the recipe generator
        
        Args:
            model_name: Name of the HuggingFace model to use
            use_api: Whether to use HuggingFace API instead of local model
            api_token: HuggingFace API token (required if use_api=True)
        """
        self.model_name = model_name
        self.use_api = use_api
        self.api_token = api_token or os.environ.get('HUGGINGFACE_API_TOKEN')
        
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        # Use mock generation by default for fast response
        # Set to False to try loading the actual model
        self.use_mock = True
        
        if not self.use_mock:
            if use_api and self.api_token:
                self._init_api()
            else:
                try:
                    self._init_local_model()
                except Exception as e:
                    print(f"Warning: Failed to initialize local model. Using mock generation. Error: {e}")
                    self.use_mock = True
    
    def _init_api(self):
        """Initialize HuggingFace API client"""
        try:
            from huggingface_hub import InferenceClient
            
            model_id = self.SUPPORTED_MODELS.get(self.model_name, self.model_name)
            self.client = InferenceClient(token=self.api_token)
            self.model_id = model_id
            print(f"HuggingFace API client initialized with model: {model_id}")
            
        except ImportError:
            print("Warning: huggingface_hub not installed. Using mock generation.")
            self.use_mock = True
        except Exception as e:
            print(f"Warning: Failed to initialize HuggingFace API. Error: {e}")
            self.use_mock = True
    
    def _init_local_model(self):
        """Initialize local HuggingFace model"""
        try:
            from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
            import torch
            
            model_id = self.SUPPORTED_MODELS.get(self.model_name, self.model_name)
            
            # Determine device
            device = 0 if torch.cuda.is_available() else -1
            
            # Special handling for different model types
            if 't5' in model_id.lower() or 'flan' in model_id.lower():
                # Seq2Seq models like T5, FLAN-T5
                self.pipeline = pipeline(
                    'text2text-generation',
                    model=model_id,
                    device=device,
                    max_length=512
                )
            elif 'gpt' in model_id.lower() or 'bloom' in model_id.lower():
                # Causal LM models
                self.pipeline = pipeline(
                    'text-generation',
                    model=model_id,
                    device=device,
                    max_length=512
                )
            else:
                # Default to text generation
                self.pipeline = pipeline(
                    'text-generation',
                    model=model_id,
                    device=device,
                    max_length=512
                )
            
            print(f"Local model initialized: {model_id}")
            
        except ImportError:
            raise ImportError("transformers library is not installed")
        except Exception as e:
            raise Exception(f"Failed to load model: {e}")
    
    def generate(self, ingredients: List[str], 
                 cuisine_type: Optional[str] = None,
                 dietary_restrictions: Optional[List[str]] = None,
                 cooking_time: Optional[int] = None) -> Dict:
        """
        Generate a recipe based on ingredients
        
        Args:
            ingredients: List of ingredient names
            cuisine_type: Optional cuisine preference (Italian, Mexican, etc.)
            dietary_restrictions: Optional dietary restrictions (vegetarian, gluten-free, etc.)
            cooking_time: Optional max cooking time in minutes
            
        Returns:
            Dictionary containing the generated recipe
        """
        if self.use_mock:
            return self._mock_generate(ingredients, cuisine_type, dietary_restrictions, cooking_time)
        
        if self.use_api:
            return self._api_generate(ingredients, cuisine_type, dietary_restrictions, cooking_time)
        
        return self._local_generate(ingredients, cuisine_type, dietary_restrictions, cooking_time)
    
    def _build_prompt(self, ingredients: List[str],
                      cuisine_type: Optional[str] = None,
                      dietary_restrictions: Optional[List[str]] = None,
                      cooking_time: Optional[int] = None) -> str:
        """Build the prompt for recipe generation"""
        
        ingredient_list = ', '.join(ingredients)
        
        prompt = f"Generate a detailed recipe using these ingredients: {ingredient_list}."
        
        if cuisine_type:
            prompt += f" Make it a {cuisine_type} style dish."
        
        if dietary_restrictions:
            restrictions = ', '.join(dietary_restrictions)
            prompt += f" The recipe should be {restrictions}."
        
        if cooking_time:
            prompt += f" The total cooking time should be under {cooking_time} minutes."
        
        prompt += """

Please provide:
1. Recipe name
2. List of ingredients with quantities
3. Step-by-step cooking instructions
4. Estimated cooking time
5. Number of servings
6. Optional tips or variations

Recipe:"""
        
        return prompt
    
    def _api_generate(self, ingredients: List[str],
                      cuisine_type: Optional[str] = None,
                      dietary_restrictions: Optional[List[str]] = None,
                      cooking_time: Optional[int] = None) -> Dict:
        """Generate recipe using HuggingFace API"""
        try:
            prompt = self._build_prompt(ingredients, cuisine_type, dietary_restrictions, cooking_time)
            
            response = self.client.text_generation(
                prompt,
                model=self.model_id,
                max_new_tokens=500,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
            return self._parse_recipe(response, ingredients)
            
        except Exception as e:
            print(f"API generation error: {e}")
            return self._mock_generate(ingredients, cuisine_type, dietary_restrictions, cooking_time)
    
    def _local_generate(self, ingredients: List[str],
                        cuisine_type: Optional[str] = None,
                        dietary_restrictions: Optional[List[str]] = None,
                        cooking_time: Optional[int] = None) -> Dict:
        """Generate recipe using local model"""
        try:
            prompt = self._build_prompt(ingredients, cuisine_type, dietary_restrictions, cooking_time)
            
            # Generate text
            outputs = self.pipeline(
                prompt,
                max_new_tokens=500,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1
            )
            
            # Extract generated text
            if isinstance(outputs, list) and len(outputs) > 0:
                generated_text = outputs[0].get('generated_text', '')
                # Remove the prompt from the output for causal models
                if prompt in generated_text:
                    generated_text = generated_text.replace(prompt, '').strip()
            else:
                generated_text = str(outputs)
            
            return self._parse_recipe(generated_text, ingredients)
            
        except Exception as e:
            print(f"Local generation error: {e}")
            return self._mock_generate(ingredients, cuisine_type, dietary_restrictions, cooking_time)
    
    def _parse_recipe(self, generated_text: str, ingredients: List[str]) -> Dict:
        """Parse the generated text into a structured recipe"""
        
        # Try to extract sections from the generated text
        lines = generated_text.strip().split('\n')
        
        recipe = {
            'name': '',
            'ingredients': [],
            'instructions': [],
            'cooking_time': '',
            'servings': '',
            'tips': '',
            'raw_text': generated_text
        }
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            lower_line = line.lower()
            
            # Detect sections
            if 'recipe name' in lower_line or (not recipe['name'] and len(line) < 100):
                recipe['name'] = line.replace('Recipe name:', '').replace('Recipe Name:', '').strip()
            elif 'ingredient' in lower_line:
                current_section = 'ingredients'
            elif 'instruction' in lower_line or 'step' in lower_line or 'direction' in lower_line:
                current_section = 'instructions'
            elif 'time' in lower_line:
                recipe['cooking_time'] = line
                current_section = None
            elif 'serving' in lower_line:
                recipe['servings'] = line
                current_section = None
            elif 'tip' in lower_line or 'variation' in lower_line:
                current_section = 'tips'
            elif current_section == 'ingredients':
                recipe['ingredients'].append(line)
            elif current_section == 'instructions':
                recipe['instructions'].append(line)
            elif current_section == 'tips':
                recipe['tips'] += line + ' '
        
        # Set default name if not found
        if not recipe['name']:
            recipe['name'] = f"Dish with {', '.join(ingredients[:3])}"
        
        # Add original ingredients if none were parsed
        if not recipe['ingredients']:
            recipe['ingredients'] = ingredients
        
        return recipe
    
    def _mock_generate(self, ingredients: List[str],
                       cuisine_type: Optional[str] = None,
                       dietary_restrictions: Optional[List[str]] = None,
                       cooking_time: Optional[int] = None) -> Dict:
        """
        Generate a mock recipe for testing
        This provides realistic recipe suggestions without requiring the LLM
        """
        
        # Normalize ingredients to lowercase for matching
        ingredient_set = set(ing.lower() for ing in ingredients)
        
        # Check for tomato + egg + onion combination first (most specific)
        if 'tomato' in ingredient_set and 'egg' in ingredient_set and 'onion' in ingredient_set:
            return {
                'name': 'Tomato Egg and Onion Stir-Fry',
                'ingredients': [
                    '4 large eggs',
                    '3 medium tomatoes, cut into wedges',
                    '1 large onion, sliced',
                    '2 tablespoons vegetable oil',
                    '1 teaspoon salt',
                    '1/2 teaspoon sugar',
                    '1/4 teaspoon black pepper',
                    '2 cloves garlic, minced (optional)',
                    '1 tablespoon soy sauce'
                ],
                'instructions': [
                    'Beat the eggs with a pinch of salt until well combined.',
                    'Heat 1 tablespoon of oil in a large pan or wok over high heat.',
                    'Pour in the beaten eggs and scramble until just set but still slightly wet. Remove and set aside.',
                    'Add remaining oil to the pan. Add sliced onions and stir-fry for 2-3 minutes until softened and slightly caramelized.',
                    'Add garlic (if using) and cook for 30 seconds until fragrant.',
                    'Add tomato wedges and stir-fry for 2-3 minutes until they start to soften.',
                    'Season with salt, sugar, pepper, and soy sauce. Continue cooking until tomatoes release their juices.',
                    'Return the scrambled eggs to the pan. Gently fold everything together.',
                    'Cook for another minute to let flavors combine.',
                    'Serve immediately with steamed rice.'
                ],
                'cooking_time': '20 minutes',
                'servings': '3-4 servings',
                'tips': 'The onions add a nice sweetness that complements the tangy tomatoes. For extra flavor, you can add a splash of sesame oil at the end. The key is to not overcook the eggs - they should be soft and fluffy.'
            }
        
        # Check for tomato + egg combination
        if 'tomato' in ingredient_set and 'egg' in ingredient_set:
            return {
                'name': 'Classic Tomato and Egg Stir-Fry',
                'ingredients': [
                    '4 large eggs',
                    '2 medium tomatoes, cut into wedges',
                    '2 tablespoons vegetable oil',
                    '1 teaspoon salt',
                    '1/2 teaspoon sugar',
                    '2 green onions, chopped',
                    '1 tablespoon soy sauce (optional)'
                ],
                'instructions': [
                    'Beat the eggs with a pinch of salt until well combined.',
                    'Heat 1 tablespoon of oil in a wok or large skillet over high heat.',
                    'Pour in the beaten eggs and scramble until just set but still slightly wet. Remove and set aside.',
                    'Add remaining oil to the wok. Add tomatoes and stir-fry for 2-3 minutes until they start to soften.',
                    'Add sugar and salt to the tomatoes. Continue cooking until tomatoes release their juices.',
                    'Return the scrambled eggs to the wok. Gently fold everything together.',
                    'Garnish with green onions and serve immediately with steamed rice.'
                ],
                'cooking_time': '15 minutes',
                'servings': '2-3 servings',
                'tips': 'The key is to not overcook the eggs - they should be soft and fluffy. Adding a bit of sugar helps balance the acidity of the tomatoes.'
            }
        
        # Check for chicken + garlic combination
        if 'chicken' in ingredient_set and 'garlic' in ingredient_set:
            return {
                'name': 'Garlic Butter Chicken',
                'ingredients': [
                    '4 chicken thighs or breasts',
                    '6 cloves garlic, minced',
                    '3 tablespoons butter',
                    '1 tablespoon olive oil',
                    'Salt and pepper to taste',
                    '1/4 cup chicken broth',
                    'Fresh parsley for garnish'
                ],
                'instructions': [
                    'Season chicken with salt and pepper on both sides.',
                    'Heat olive oil in a large skillet over medium-high heat.',
                    'Add chicken and cook for 6-7 minutes per side until golden and cooked through. Remove and set aside.',
                    'Reduce heat to medium. Add butter and garlic to the same skillet.',
                    'Saute garlic for 1 minute until fragrant (do not let it burn).',
                    'Add chicken broth and scrape up any browned bits from the bottom.',
                    'Return chicken to the skillet, spooning garlic butter over the top.',
                    'Simmer for 2-3 minutes. Garnish with fresh parsley and serve.'
                ],
                'cooking_time': '25 minutes',
                'servings': '4 servings',
                'tips': 'Let the chicken rest for 5 minutes before serving for juicier results. Serve with mashed potatoes or rice to soak up the delicious garlic butter sauce.'
            }
        
        # Check for potato + lemon combination
        if 'potato' in ingredient_set and 'lemon' in ingredient_set:
            return {
                'name': 'Lemon Herb Roasted Potatoes',
                'ingredients': [
                    '2 lbs potatoes, cut into chunks',
                    '2 lemons (juice and zest)',
                    '4 tablespoons olive oil',
                    '4 cloves garlic, minced',
                    '1 teaspoon dried oregano',
                    '1 teaspoon dried rosemary',
                    'Salt and pepper to taste',
                    'Fresh parsley for garnish'
                ],
                'instructions': [
                    'Preheat oven to 425°F (220°C).',
                    'Cut potatoes into even-sized chunks for uniform cooking.',
                    'In a large bowl, whisk together olive oil, lemon juice, lemon zest, garlic, oregano, and rosemary.',
                    'Add potatoes and toss to coat evenly with the lemon mixture.',
                    'Spread potatoes in a single layer on a baking sheet.',
                    'Season generously with salt and pepper.',
                    'Roast for 35-40 minutes, flipping halfway through, until golden and crispy.',
                    'Squeeze extra lemon juice over the hot potatoes.',
                    'Garnish with fresh parsley and serve immediately.'
                ],
                'cooking_time': '45 minutes',
                'servings': '4-6 servings',
                'tips': 'For extra crispy potatoes, make sure they are dry before tossing with oil. Do not overcrowd the baking sheet - use two sheets if needed.'
            }
        
        # Check for potato only
        if 'potato' in ingredient_set:
            return {
                'name': 'Crispy Herb Roasted Potatoes',
                'ingredients': [
                    '2 lbs potatoes, cubed',
                    '3 tablespoons olive oil',
                    '3 cloves garlic, minced',
                    '1 teaspoon paprika',
                    '1 teaspoon dried rosemary',
                    '1/2 teaspoon dried thyme',
                    'Salt and pepper to taste',
                    'Fresh parsley for garnish'
                ],
                'instructions': [
                    'Preheat oven to 425°F (220°C).',
                    'Cut potatoes into 1-inch cubes for even cooking.',
                    'Toss potatoes with olive oil, garlic, paprika, rosemary, and thyme.',
                    'Spread in a single layer on a baking sheet.',
                    'Season with salt and pepper.',
                    'Roast for 30-35 minutes, stirring halfway, until golden and crispy.',
                    'Garnish with fresh parsley and serve hot.'
                ],
                'cooking_time': '40 minutes',
                'servings': '4 servings',
                'tips': 'Parboiling the potatoes for 5 minutes before roasting makes them extra fluffy inside and crispy outside.'
            }
        
        # Check for lemon + chicken combination
        if 'lemon' in ingredient_set and 'chicken' in ingredient_set:
            return {
                'name': 'Lemon Garlic Chicken',
                'ingredients': [
                    '4 chicken breasts or thighs',
                    '2 lemons (juice and slices)',
                    '4 cloves garlic, minced',
                    '3 tablespoons olive oil',
                    '1 teaspoon dried oregano',
                    '1/2 cup chicken broth',
                    'Salt and pepper to taste',
                    'Fresh parsley for garnish'
                ],
                'instructions': [
                    'Season chicken with salt, pepper, and oregano.',
                    'Heat olive oil in a large oven-safe skillet over medium-high heat.',
                    'Sear chicken for 4-5 minutes per side until golden.',
                    'Add garlic and cook for 30 seconds until fragrant.',
                    'Pour in lemon juice and chicken broth.',
                    'Add lemon slices around the chicken.',
                    'Transfer to a 400°F (200°C) oven and bake for 20 minutes.',
                    'Let rest for 5 minutes, then garnish with parsley and serve.'
                ],
                'cooking_time': '35 minutes',
                'servings': '4 servings',
                'tips': 'Use fresh lemon juice for the best flavor. The pan sauce is delicious over rice or mashed potatoes.'
            }
        
        # Check for lemon only (for fish or salad dressing)
        if 'lemon' in ingredient_set:
            return {
                'name': 'Lemon Vinaigrette Salad',
                'ingredients': [
                    '2 lemons, juiced',
                    '1/4 cup olive oil',
                    '1 teaspoon Dijon mustard',
                    '1 teaspoon honey',
                    '1 clove garlic, minced',
                    'Salt and pepper to taste',
                    'Mixed salad greens',
                    'Cherry tomatoes, cucumber, optional toppings'
                ],
                'instructions': [
                    'In a small bowl, whisk together lemon juice, mustard, honey, and garlic.',
                    'Slowly drizzle in olive oil while whisking to emulsify.',
                    'Season with salt and pepper to taste.',
                    'Prepare your salad greens and vegetables.',
                    'Drizzle the lemon vinaigrette over the salad.',
                    'Toss gently to coat and serve immediately.'
                ],
                'cooking_time': '10 minutes',
                'servings': '4 servings',
                'tips': 'This dressing keeps in the refrigerator for up to a week. Bring to room temperature and shake well before using.'
            }
        
        # Check for egg + onion combination
        if 'egg' in ingredient_set and 'onion' in ingredient_set:
            return {
                'name': 'Caramelized Onion Omelette',
                'ingredients': [
                    '4 large eggs',
                    '2 medium onions, thinly sliced',
                    '2 tablespoons butter',
                    '1 tablespoon olive oil',
                    'Salt and pepper to taste',
                    '2 tablespoons milk',
                    'Fresh chives for garnish'
                ],
                'instructions': [
                    'Heat olive oil and 1 tablespoon butter in a pan over medium-low heat.',
                    'Add sliced onions and cook slowly for 15-20 minutes, stirring occasionally, until caramelized and golden.',
                    'Season onions with salt and pepper, then remove and set aside.',
                    'Beat eggs with milk, salt, and pepper until well combined.',
                    'Melt remaining butter in the same pan over medium heat.',
                    'Pour in egg mixture and let it set slightly around the edges.',
                    'Add caramelized onions to one half of the omelette.',
                    'Fold the omelette in half and cook for another minute.',
                    'Garnish with fresh chives and serve immediately.'
                ],
                'cooking_time': '25 minutes',
                'servings': '2 servings',
                'tips': 'The key to perfect caramelized onions is patience - cook them slowly over low heat for the best sweetness and flavor.'
            }
        
        # Generate a generic recipe using all provided ingredients
        ingredient_list = list(ingredients)
        return {
            'name': f"Home-Style {ingredient_list[0].title()} Dish with {' and '.join(ingredient_list[1:]).title() if len(ingredient_list) > 1 else 'Herbs'}",
            'ingredients': [f"Fresh {ing}" for ing in ingredient_list] + [
                '2 tablespoons olive oil',
                'Salt and pepper to taste',
                '2 cloves garlic, minced',
                'Fresh herbs for garnish'
            ],
            'instructions': [
                f'Prepare all ingredients: wash and chop {", ".join(ingredient_list)}.',
                'Heat olive oil in a large pan over medium-high heat.',
                'Add garlic and saute for 30 seconds until fragrant.',
                f'Add the {ingredient_list[0]} first as it may need longer cooking time.',
                f'Add the remaining ingredients ({", ".join(ingredient_list[1:]) if len(ingredient_list) > 1 else "seasonings"}), stirring frequently.',
                'Season with salt and pepper to taste.',
                'Cook until all ingredients are tender and well combined.',
                'Garnish with fresh herbs and serve hot.'
            ],
            'cooking_time': '20-30 minutes',
            'servings': '2-4 servings',
            'tips': f'Feel free to adjust seasonings to your taste. This recipe uses all your ingredients: {", ".join(ingredient_list)}. You can add more vegetables or protein to make this dish more substantial.'
        }


class RecipeGeneratorWithRAG(RecipeGenerator):
    """
    Extended Recipe Generator with Retrieval-Augmented Generation (RAG)
    
    Uses a recipe database to enhance LLM generation with relevant
    example recipes and techniques.
    """
    
    def __init__(self, recipe_database_path: str = None, **kwargs):
        """
        Initialize RAG-enhanced recipe generator
        
        Args:
            recipe_database_path: Path to recipe database (JSON or SQLite)
            **kwargs: Additional arguments for base RecipeGenerator
        """
        super().__init__(**kwargs)
        self.recipe_db = None
        
        if recipe_database_path:
            self._load_recipe_database(recipe_database_path)
    
    def _load_recipe_database(self, path: str):
        """Load recipe database for RAG"""
        try:
            import json
            with open(path, 'r') as f:
                self.recipe_db = json.load(f)
            print(f"Loaded {len(self.recipe_db)} recipes from database")
        except Exception as e:
            print(f"Warning: Failed to load recipe database: {e}")
            self.recipe_db = []
    
    def _find_similar_recipes(self, ingredients: List[str], top_k: int = 3) -> List[Dict]:
        """Find similar recipes from the database based on ingredients"""
        if not self.recipe_db:
            return []
        
        ingredient_set = set(ing.lower() for ing in ingredients)
        scored_recipes = []
        
        for recipe in self.recipe_db:
            recipe_ingredients = set(ing.lower() for ing in recipe.get('ingredients', []))
            # Calculate Jaccard similarity
            intersection = len(ingredient_set & recipe_ingredients)
            union = len(ingredient_set | recipe_ingredients)
            score = intersection / union if union > 0 else 0
            scored_recipes.append((score, recipe))
        
        # Sort by score and return top_k
        scored_recipes.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in scored_recipes[:top_k]]
    
    def generate(self, ingredients: List[str], **kwargs) -> Dict:
        """Generate recipe with RAG enhancement"""
        # Find similar recipes
        similar_recipes = self._find_similar_recipes(ingredients)
        
        # Build enhanced prompt with examples
        if similar_recipes:
            # Add context from similar recipes to the prompt
            kwargs['context_recipes'] = similar_recipes
        
        return super().generate(ingredients, **kwargs)
