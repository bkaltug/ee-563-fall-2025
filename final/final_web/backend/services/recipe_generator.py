import os
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class RecipeGenerator:
    
    SUPPORTED_MODELS = {
        'gpt2': 'gpt2',
        'gpt2-medium': 'gpt2-medium',
        'distilgpt2': 'distilgpt2',
        'bloom': 'bigscience/bloom-560m',
        'flan-t5': 'google/flan-t5-base',
        'flan-t5-large': 'google/flan-t5-large',
        'mistral': 'mistralai/Mistral-7B-Instruct-v0.1',
        'llama2': 'meta-llama/Llama-2-7b-chat-hf',
        'recipe-nlg': 'flax-community/t5-recipe-generation',
    }
    
    DEFAULT_QUANTITIES = {
        # Proteins
        'meat': '1 lb ground beef',
        'beef': '1 lb beef stew meat, cubed',
        'chicken': '1 lb boneless skinless chicken breast',
        'pork': '1 lb pork tenderloin, sliced',
        'fish': '1 lb white fish fillets',
        'shrimp': '1 lb large shrimp, peeled and deveined',
        'bacon': '6 strips thick-cut bacon',
        'sausage': '4 Italian sausage links',
        'turkey': '1 lb ground turkey',
        'lamb': '1 lb lamb shoulder, cubed',
        # Dairy
        'cheese': '1 cup shredded cheddar cheese',
        'milk': '1 cup whole milk',
        'butter': '4 tbsp unsalted butter',
        'cream': '1 cup heavy whipping cream',
        'egg': '4 large eggs',
        'eggs': '4 large eggs',
        'yogurt': '1 cup plain yogurt',
        # Vegetables
        'tomato': '3 medium ripe tomatoes, diced',
        'lettuce': '1 head romaine lettuce, chopped',
        'onion': '1 large yellow onion, diced',
        'garlic': '4 cloves garlic, minced',
        'potato': '4 medium russet potatoes, cubed',
        'carrot': '3 large carrots, sliced',
        'pepper': '2 bell peppers, diced',
        'mushroom': '8 oz cremini mushrooms, sliced',
        'broccoli': '2 cups broccoli florets',
        'spinach': '4 cups fresh baby spinach',
        'cucumber': '1 English cucumber, sliced',
        'celery': '3 stalks celery, chopped',
        'corn': '2 cups fresh corn kernels',
        'peas': '1 cup frozen peas',
        'beans': '1 can (15 oz) black beans, drained',
        'zucchini': '2 medium zucchini, sliced',
        'cabbage': '1/2 head green cabbage, shredded',
        'cauliflower': '1 head cauliflower, cut into florets',
        'asparagus': '1 bunch asparagus, trimmed',
        'green beans': '1 lb fresh green beans, trimmed',
        # Grains/Bread
        'bread': '4 slices sandwich bread',
        'rice': '2 cups long grain white rice',
        'pasta': '1 lb spaghetti pasta',
        'noodles': '8 oz egg noodles',
        'flour': '2 cups all-purpose flour',
        'oats': '1 cup old-fashioned rolled oats',
        # Seasonings & Others
        'salt': '1 tsp kosher salt',
        'oil': '3 tbsp extra virgin olive oil',
        'sugar': '2 tbsp granulated sugar',
        'honey': '2 tbsp honey',
        'soy sauce': '3 tbsp soy sauce',
        'vinegar': '2 tbsp white wine vinegar',
        'lemon': '2 fresh lemons, juiced',
        'lime': '2 fresh limes, juiced',
        'ginger': '1 tbsp fresh ginger, minced',
    }
    
    def __init__(self, model_name: str = 'recipe-nlg', use_api: bool = False, api_token: str = None, lazy_load: bool = True):
        self.model_name = model_name
        self.use_api = use_api
        self.api_token = api_token or os.environ.get('HUGGINGFACE_API_TOKEN')
        
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.use_mock = False
        self._initialized = False
        
        # Lazy loading: only initialize when actually needed (on first generate call)
        if not lazy_load:
            self._ensure_initialized()
    
    def _ensure_initialized(self):
        if self._initialized:
            return
        
        if self.use_api and self.api_token:
            self._init_api()
        else:
            self._init_local_model()
        
        self._initialized = True
    
    def _init_api(self):
        from huggingface_hub import InferenceClient
        
        model_id = self.SUPPORTED_MODELS.get(self.model_name, self.model_name)
        self.client = InferenceClient(token=self.api_token)
        self.model_id = model_id
        print(f"HuggingFace API client initialized with model: {model_id}")
    
    def _init_local_model(self):
        from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
        import torch
        
        model_id = self.SUPPORTED_MODELS.get(self.model_name, self.model_name)
        
        # Determine device
        device = 0 if torch.cuda.is_available() else -1
        device_name = "GPU" if device == 0 else "CPU"
        print(f"Initializing model on {device_name}...")
        
        if 't5' in model_id.lower() or 'flan' in model_id.lower():
            self.pipeline = pipeline(
                'text2text-generation',
                model=model_id,
                device=device,
                max_length=512
            )
        elif 'gpt' in model_id.lower() or 'bloom' in model_id.lower():
            self.pipeline = pipeline(
                'text-generation',
                model=model_id,
                device=device,
                max_length=512
            )
        else:
            self.pipeline = pipeline(
                'text-generation',
                model=model_id,
                device=device,
                max_length=512
            )
        
        print(f"✓ Local model initialized: {model_id}")
    
    def generate(self, ingredients: List[str], 
                 cuisine_type: Optional[str] = None,
                 dietary_restrictions: Optional[List[str]] = None,
                 cooking_time: Optional[int] = None) -> Dict:
        self._ensure_initialized()
        
        if self.use_api:
            return self._api_generate(ingredients, cuisine_type, dietary_restrictions, cooking_time)
        
        return self._local_generate(ingredients, cuisine_type, dietary_restrictions, cooking_time)
    
    def _build_prompt(self, ingredients: List[str],
                      cuisine_type: Optional[str] = None,
                      dietary_restrictions: Optional[List[str]] = None,
                      cooking_time: Optional[int] = None) -> str:
        ingredient_with_quantities = []
        for ing in ingredients:
            ing_lower = ing.lower().strip()
            if ing_lower in self.DEFAULT_QUANTITIES:
                ingredient_with_quantities.append(self.DEFAULT_QUANTITIES[ing_lower])
            else:
                ingredient_with_quantities.append(f"1 cup {ing}")
        
        ingredient_list = ', '.join(ingredient_with_quantities)
        
        if self.model_name == 'recipe-nlg':
            prompt = f"items: {ingredient_list}"
            return prompt
        
        prompt = f"Write a complete cooking recipe using {ingredient_list}."
        
        if cuisine_type:
            prompt = f"Write a {cuisine_type} recipe using {ingredient_list}."
        
        if dietary_restrictions:
            restrictions = ', '.join(dietary_restrictions)
            prompt += f" Make it {restrictions}."
        
        if cooking_time:
            prompt += f" Cooking time under {cooking_time} minutes."
        
        prompt += " Include the recipe name, ingredient amounts, and step-by-step instructions."
        
        return prompt
    
    def _api_generate(self, ingredients: List[str],
                      cuisine_type: Optional[str] = None,
                      dietary_restrictions: Optional[List[str]] = None,
                      cooking_time: Optional[int] = None) -> Dict:
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
    
    def _local_generate(self, ingredients: List[str],
                        cuisine_type: Optional[str] = None,
                        dietary_restrictions: Optional[List[str]] = None,
                        cooking_time: Optional[int] = None) -> Dict:
        prompt = self._build_prompt(ingredients, cuisine_type, dietary_restrictions, cooking_time)
        
        max_attempts = 2
        for attempt in range(max_attempts):
            outputs = self.pipeline(
                prompt,
                max_new_tokens=512,
                temperature=0.7 + (attempt * 0.1),
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1,
                repetition_penalty=1.2 + (attempt * 0.1),
                no_repeat_ngram_size=3
            )
            
            if isinstance(outputs, list) and len(outputs) > 0:
                generated_text = outputs[0].get('generated_text', '')
                if prompt in generated_text:
                    generated_text = generated_text.replace(prompt, '').strip()
            else:
                generated_text = str(outputs)
            
            if self._is_garbage_output(generated_text):
                if attempt < max_attempts - 1:
                    print(f"Detected garbage output, retrying (attempt {attempt + 2})...")
                    continue
            break
        
        # Parse the recipe and enhance if needed
        recipe = self._parse_recipe(generated_text, ingredients)
        
        # Ensure we have good formatted ingredients with quantities
        if not recipe['ingredients'] or len(recipe['ingredients']) < len(ingredients):
            recipe['ingredients'] = self._format_ingredients_with_quantities(ingredients)
        
        if not recipe['instructions']:
            recipe['instructions'] = self._generate_basic_instructions(ingredients)
        
        if not recipe['cooking_time']:
            recipe['cooking_time'] = self._estimate_cooking_time(ingredients)
        if not recipe['servings']:
            recipe['servings'] = '4 servings'
        
        return recipe
    
    def _is_garbage_output(self, text: str) -> bool:
        import re
        
        words = text.lower().split()
        if len(words) < 20:
            return False
        
        from collections import Counter
        word_counts = Counter(words)
        most_common_count = word_counts.most_common(1)[0][1] if word_counts else 0
        
        if most_common_count > len(words) * 0.2:
            return True
        
        pattern = r'(\b\w+\s+\w+\s+\w+\b)(?:.*?\1){5,}'
        if re.search(pattern, text.lower()):
            return True
        
        return False
    
    def _generate_basic_instructions(self, ingredients: List[str]) -> List[str]:
        instructions = []
        ing_str = ", ".join(ingredients[:-1]) + " and " + ingredients[-1] if len(ingredients) > 1 else ingredients[0]
        
        instructions.append(f"Step 1: Gather and prepare all ingredients: {ing_str}.")
        instructions.append("Step 2: Wash and chop vegetables as needed.")
        
        proteins = {'chicken', 'beef', 'pork', 'fish', 'shrimp', 'meat', 'bacon', 'sausage'}
        if any(p in ing.lower() for ing in ingredients for p in proteins):
            instructions.append("Step 3: Cook the protein in a pan over medium-high heat until done.")
            instructions.append("Step 4: Combine all ingredients together.")
            instructions.append("Step 5: Season to taste with salt and pepper. Serve warm.")
        else:
            instructions.append("Step 3: Combine all ingredients together in a bowl or pan.")
            instructions.append("Step 4: Cook or mix as appropriate for the dish.")
            instructions.append("Step 5: Season to taste and serve.")
        
        return instructions

    def _format_ingredients_with_quantities(self, ingredients: List[str]) -> List[str]:
        formatted = []
        for ing in ingredients:
            ing_lower = ing.lower().strip()
            if ing_lower in self.DEFAULT_QUANTITIES:
                formatted.append(self.DEFAULT_QUANTITIES[ing_lower])
            else:
                formatted.append(f"1 cup {ing}")
        return formatted
    
    def _estimate_cooking_time(self, ingredients: List[str]) -> str:
        ing_set = set(ing.lower() for ing in ingredients)
        
        if ing_set & {'beef', 'pork', 'lamb', 'chicken'}:
            return '30-45 minutes'
        elif ing_set & {'fish', 'shrimp'}:
            return '15-20 minutes'
        elif ing_set & {'pasta', 'rice', 'noodles'}:
            return '20-25 minutes'
        elif ing_set & {'egg', 'eggs', 'bread'}:
            return '10-15 minutes'
        else:
            return '20-30 minutes'
    
    def _parse_recipe(self, generated_text: str, ingredients: List[str]) -> Dict:
        
        recipe = {
            'name': '',
            'ingredients': [],
            'instructions': [],
            'cooking_time': '',
            'servings': '',
            'tips': '',
            'raw_text': generated_text
        }
        
        text = generated_text.strip()
        
        if 'title:' in text.lower() and 'directions:' in text.lower():
            import re
            
            title_match = text.lower().find('title:')
            ingredients_match = text.lower().find('ingredients:')
            directions_match = text.lower().find('directions:')
            
            if title_match != -1 and ingredients_match != -1:
                name = text[title_match + 6:ingredients_match].strip()
                recipe['name'] = name.title()
            
            if ingredients_match != -1 and directions_match != -1:
                ingredients_text = text[ingredients_match + 12:directions_match].strip()
                ingredients_text = re.sub(r'^ingredients\s*', '', ingredients_text, flags=re.IGNORECASE)
                
                ingredient_pattern = r'(\d+(?:/\d+)?\s*(?:cup|cups|c\.|tbsp|tsp|teaspoon|tablespoon|lb|lbs|pound|pounds|oz|ounce|ounces|slice|slices|piece|pieces|clove|cloves|head|heads|can|cans|pkg|package|bunch|pinch)?\s*\.?\s*[^\d]+?)(?=\d|$)'
                parsed_ingredients = re.findall(ingredient_pattern, ingredients_text, re.IGNORECASE)
                
                if parsed_ingredients:
                    seen = set()
                    for item in parsed_ingredients:
                        item = item.strip().strip(',').strip('.')
                        if item and len(item) > 2 and item.lower() not in seen:
                            words = item.split()
                            unit_words = {'cup', 'cups', 'c', 'tbsp', 'tsp', 'teaspoon', 'tablespoon', 
                                         'lb', 'lbs', 'pound', 'pounds', 'oz', 'ounce', 'ounces',
                                         'slice', 'slices', 'piece', 'pieces', 'inch', 'small', 
                                         'medium', 'large', 'clove', 'cloves', 'head', 'heads', 
                                         'can', 'cans', 'pkg', 'package', 'bunch', 'pinch',
                                         'chopped', 'diced', 'sliced', 'minced', 'crushed', 'fresh',
                                         'dried', 'ground', 'whole', 'cut', 'cubed'}
                            has_ingredient_name = any(
                                len(w) >= 3 and w.lower() not in unit_words and not w.replace('/', '').replace('.', '').isdigit()
                                for w in words
                            )
                            if has_ingredient_name:
                                seen.add(item.lower())
                                recipe['ingredients'].append(item)
                else:
                    seen = set()
                    for item in ingredients_text.replace(' . ', ', ').split(','):
                        item = item.strip().strip('.')
                        if item and len(item) > 2 and item.lower() not in seen:
                            seen.add(item.lower())
                            recipe['ingredients'].append(item)
            
            if directions_match != -1:
                directions_text = text[directions_match + 11:].strip()
                directions_text = re.sub(r'^directions\s*', '', directions_text, flags=re.IGNORECASE)
                
                steps = re.split(r'(?:\d+\.\s*)', directions_text)
                
                if len(steps) <= 1 and directions_text:
                    steps = re.split(r'\.\s+', directions_text)
                
                step_num = 1
                for step in steps:
                    step = step.strip()
                    if step and len(step) > 5:
                        if not step.endswith('.'):
                            step += '.'
                        step = step[0].upper() + step[1:] if len(step) > 1 else step.upper()
                        recipe['instructions'].append(f"Step {step_num}: {step}")
                        step_num += 1
            
            if directions_text:
                time_match = re.search(r'(\d+)\s*(?:to\s*(\d+)\s*)?minutes', directions_text.lower())
                if time_match:
                    if time_match.group(2):
                        recipe['cooking_time'] = f"{time_match.group(1)}-{time_match.group(2)} minutes"
                    else:
                        recipe['cooking_time'] = f"{time_match.group(1)} minutes"
                        
                hour_match = re.search(r'(\d+)\s*hours?', directions_text.lower())
                if hour_match:
                    recipe['cooking_time'] = f"{hour_match.group(1)} hour(s)"
            
            if not recipe['cooking_time']:
                recipe['cooking_time'] = "15-20 minutes"
            if not recipe['servings']:
                recipe['servings'] = "2-4 servings"
            
            return recipe
        
        lines = text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            lower_line = line.lower()
            
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
        
        if not recipe['name']:
            recipe['name'] = f"Dish with {', '.join(ingredients[:3])}"
        
        if not recipe['ingredients']:
            recipe['ingredients'] = ingredients
        
        return recipe
    
    def _mock_generate(self, ingredients: List[str],
                       cuisine_type: Optional[str] = None,
                       dietary_restrictions: Optional[List[str]] = None,
                       cooking_time: Optional[int] = None) -> Dict:
        ingredient_set = set(ing.lower() for ing in ingredients)
        
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
 
    
    def __init__(self, recipe_database_path: str = None, **kwargs):

        super().__init__(**kwargs)
        self.recipe_db = None
        
        if recipe_database_path:
            self._load_recipe_database(recipe_database_path)
    
    def _load_recipe_database(self, path: str):
        try:
            import json
            with open(path, 'r') as f:
                self.recipe_db = json.load(f)
            print(f"Loaded {len(self.recipe_db)} recipes from database")
        except Exception as e:
            print(f"Warning: Failed to load recipe database: {e}")
            self.recipe_db = []
    
    def _find_similar_recipes(self, ingredients: List[str], top_k: int = 3) -> List[Dict]:
        if not self.recipe_db:
            return []
        
        ingredient_set = set(ing.lower() for ing in ingredients)
        scored_recipes = []
        
        for recipe in self.recipe_db:
            recipe_ingredients = set(ing.lower() for ing in recipe.get('ingredients', []))
            intersection = len(ingredient_set & recipe_ingredients)
            union = len(ingredient_set | recipe_ingredients)
            score = intersection / union if union > 0 else 0
            scored_recipes.append((score, recipe))
        
        scored_recipes.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in scored_recipes[:top_k]]
    
    def generate(self, ingredients: List[str], **kwargs) -> Dict:
        similar_recipes = self._find_similar_recipes(ingredients)
        
        if similar_recipes:
            kwargs['context_recipes'] = similar_recipes
        
        return super().generate(ingredients, **kwargs)
