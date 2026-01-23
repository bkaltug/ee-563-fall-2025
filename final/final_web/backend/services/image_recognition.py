import os
import numpy as np
from typing import List, Dict
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


class FoodDetector:
    
    # Comprehensive list of common food ingredients
    FOOD_INGREDIENTS = [
        # Vegetables
        'tomato', 'onion', 'garlic', 'potato', 'carrot', 'broccoli', 'spinach',
        'lettuce', 'cabbage', 'cucumber', 'bell pepper', 'zucchini', 'eggplant',
        'mushroom', 'celery', 'asparagus', 'green beans', 'peas', 'corn',
        'cauliflower', 'kale', 'leek', 'radish', 'beetroot', 'pumpkin',
        
        # Fruits
        'apple', 'banana', 'orange', 'lemon', 'lime', 'strawberry', 'blueberry',
        'grape', 'watermelon', 'mango', 'pineapple', 'peach', 'pear', 'cherry',
        'avocado', 'kiwi', 'papaya', 'pomegranate', 'fig', 'coconut',
        
        # Proteins
        'egg', 'chicken', 'beef', 'pork', 'fish', 'salmon', 'shrimp', 'tuna',
        'lamb', 'turkey', 'bacon', 'sausage', 'ham', 'steak', 'ground meat',
        'tofu', 'tempeh',
        
        # Dairy
        'cheese', 'milk', 'butter', 'cream', 'yogurt', 'mozzarella', 'parmesan',
        'cheddar cheese', 'cream cheese', 'sour cream',
        
        # Grains & Carbs
        'bread', 'rice', 'pasta', 'noodles', 'flour', 'oats', 'quinoa',
        'tortilla', 'pita bread', 'bagel', 'croissant',
        
        # Herbs & Spices
        'basil', 'parsley', 'cilantro', 'mint', 'rosemary', 'thyme', 'oregano',
        'ginger', 'chili pepper', 'jalapeno',
        
        # Pantry Items
        'olive oil', 'soy sauce', 'vinegar', 'honey', 'sugar', 'salt',
        'black pepper', 'paprika', 'cinnamon', 'nuts', 'almonds', 'peanuts',
        
        # Other common items
        'chocolate', 'coffee beans', 'tea leaves', 'beans', 'lentils', 'chickpeas',
    ]
    
    TARGET_SIZE = (224, 224)
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model = None
        self.processor = None
        self.device = None
        self.use_fallback = False
        self.model_name = model_name
        
        try:
            self._init_clip()
        except Exception as e:
            print(f"Could not initialize CLIP model: {e}")
            print("Using fallback color-based detection...")
            self.use_fallback = True
    
    def _init_clip(self):
        import torch
        from transformers import CLIPProcessor, CLIPModel
        
        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing CLIP on device: {self.device}")
        
        # Load CLIP model and processor
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Pre-compute text embeddings for all food ingredients
        self._precompute_text_embeddings()
        
        print(f"CLIP model initialized successfully! Can recognize {len(self.FOOD_INGREDIENTS)} food ingredients.")
    
    def _precompute_text_embeddings(self):
        import torch
        
        # Create text prompts for each ingredient
        text_prompts = [f"a photo of {ingredient}" for ingredient in self.FOOD_INGREDIENTS]
        
        # Process text
        text_inputs = self.processor(
            text=text_prompts,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Compute text embeddings
        with torch.no_grad():
            self.text_embeddings = self.model.get_text_features(**text_inputs)
            self.text_embeddings = self.text_embeddings / self.text_embeddings.norm(dim=-1, keepdim=True)
    
    def _preprocess_image(self, image_path: str) -> Image.Image:
        # Open image
        img = Image.open(image_path)
        
        # Convert to RGB (in case of RGBA or other formats)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Get original dimensions
        orig_width, orig_height = img.size
        print(f"Original image size: {orig_width}x{orig_height}")
        
        # Calculate aspect ratio preserving resize
        aspect = orig_width / orig_height
        
        if aspect > 1:
            # Landscape
            new_width = self.TARGET_SIZE[0]
            new_height = int(new_width / aspect)
        else:
            # Portrait or square
            new_height = self.TARGET_SIZE[1]
            new_width = int(new_height * aspect)
        
        # Ensure minimum dimensions
        new_width = max(new_width, 1)
        new_height = max(new_height, 1)
        
        # Resize with high quality
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create a square image with padding if needed
        square_img = Image.new('RGB', self.TARGET_SIZE, (128, 128, 128))
        paste_x = (self.TARGET_SIZE[0] - new_width) // 2
        paste_y = (self.TARGET_SIZE[1] - new_height) // 2
        square_img.paste(img, (paste_x, paste_y))
        
        print(f"Preprocessed image size: {self.TARGET_SIZE[0]}x{self.TARGET_SIZE[1]}")
        
        return square_img
    
    def detect(self, image_path: str, confidence_threshold: float = 0.10, max_ingredients: int = 10) -> List[Dict]:
        if self.use_fallback:
            return self._fallback_detect(image_path)
        
        return self._clip_detect(image_path, confidence_threshold, max_ingredients)
    
    def _clip_detect(self, image_path: str, confidence_threshold: float, max_ingredients: int) -> List[Dict]:
        import torch
        
        try:
            # Preprocess image
            image = self._preprocess_image(image_path)
            
            # Process image for CLIP
            image_inputs = self.processor(
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Get image embedding
            with torch.no_grad():
                image_embedding = self.model.get_image_features(**image_inputs)
                image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
            
            # Compute similarity scores with all food ingredients
            similarity = (image_embedding @ self.text_embeddings.T).squeeze(0)
            
            # Convert to probabilities using softmax
            probs = torch.softmax(similarity * 100, dim=0)  # Temperature scaling
            
            # Get top ingredients
            top_probs, top_indices = probs.topk(max_ingredients * 2)  # Get more to filter
            
            # Filter and format results
            ingredients = []
            seen_base_ingredients = set()
            
            for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
                ingredient_name = self.FOOD_INGREDIENTS[idx]
                confidence = float(prob)
                
                # Skip if below threshold
                if confidence < confidence_threshold:
                    continue
                
                # Avoid duplicates (e.g., "cheese" and "cheddar cheese")
                base_name = ingredient_name.split()[0] if ' ' in ingredient_name else ingredient_name
                if base_name in seen_base_ingredients:
                    continue
                seen_base_ingredients.add(base_name)
                
                ingredients.append({
                    'name': ingredient_name,
                    'confidence': round(confidence, 2)
                })
                
                if len(ingredients) >= max_ingredients:
                    break
            
            # If no ingredients found above threshold, return top few anyway
            if not ingredients:
                for prob, idx in zip(top_probs[:5].cpu().numpy(), top_indices[:5].cpu().numpy()):
                    ingredients.append({
                        'name': self.FOOD_INGREDIENTS[idx],
                        'confidence': round(float(prob), 2)
                    })
            
            print(f"CLIP detected {len(ingredients)} ingredients")
            return ingredients
            
        except Exception as e:
            print(f"Error during CLIP inference: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_detect(image_path)
    
    def _fallback_detect(self, image_path: str) -> List[Dict]:
        try:
            # Open and resize image
            img = Image.open(image_path).convert('RGB')
            img_small = img.resize((100, 100))
            img_array = np.array(img_small)
            pixels = img_array.reshape(-1, 3)
            
            detected = []
            detected_names = set()
            total_pixels = len(pixels)
            
            # Color-based detection rules
            detections = [
                # (name, mask_func, threshold)
                ('tomato', lambda p: (p[:, 0] > 150) & (p[:, 1] < 100) & (p[:, 2] < 100), 0.03),
                ('egg', lambda p: ((p[:, 0] > 200) & (p[:, 1] > 190) & (p[:, 2] > 170)) | 
                                  ((p[:, 0] > 180) & (p[:, 1] > 130) & (p[:, 2] < 120)), 0.04),
                ('lemon', lambda p: (p[:, 0] > 180) & (p[:, 1] > 160) & (p[:, 2] < 120), 0.03),
                ('carrot', lambda p: (p[:, 0] > 180) & (p[:, 1] > 80) & (p[:, 1] < 160) & (p[:, 2] < 100), 0.03),
                ('potato', lambda p: (p[:, 0] > 120) & (p[:, 0] < 220) & (p[:, 1] > 80) & (p[:, 1] < 190) & (p[:, 2] > 50) & (p[:, 2] < 160), 0.05),
                ('onion', lambda p: (p[:, 0] > 160) & (p[:, 0] < 230) & (p[:, 1] > 130) & (p[:, 1] < 200) & (p[:, 2] > 60) & (p[:, 2] < 150), 0.04),
                ('lettuce', lambda p: (p[:, 1] > p[:, 0] + 20) & (p[:, 1] > p[:, 2] + 20) & (p[:, 1] > 70), 0.04),
                ('chicken', lambda p: (p[:, 0] > 170) & (p[:, 0] < 240) & (p[:, 1] > 130) & (p[:, 1] < 200) & (p[:, 2] > 120) & (p[:, 2] < 190), 0.06),
            ]
            
            for name, mask_func, threshold in detections:
                mask = mask_func(pixels)
                pct = np.sum(mask) / total_pixels
                if pct > threshold:
                    confidence = min(0.85, 0.60 + pct * 2)
                    detected.append({'name': name, 'confidence': round(confidence, 2)})
                    detected_names.add(name)
            
            if not detected:
                detected = [{'name': 'unknown ingredient', 'confidence': 0.50}]
            
            return sorted(detected, key=lambda x: x['confidence'], reverse=True)
            
        except Exception as e:
            print(f"Error in fallback detection: {e}")
            return [{'name': 'unknown ingredient', 'confidence': 0.50}]
