from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import uuid

from services.image_recognition import FoodDetector
from services.recipe_generator import RecipeGenerator

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize services
food_detector = None
recipe_generator = None

def get_food_detector():
    global food_detector
    if food_detector is None:
        food_detector = FoodDetector()
    return food_detector

def get_recipe_generator():
    global recipe_generator
    if recipe_generator is None:
        recipe_generator = RecipeGenerator()
    return recipe_generator

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('../frontend', path)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Food Recipe API is running'
    })

@app.route('/api/detect', methods=['POST'])
def detect_ingredients():
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Use PNG, JPG, JPEG, or WEBP'}), 400
    
    try:
        # Save uploaded file
        filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Detect ingredients
        detector = get_food_detector()
        ingredients = detector.detect(filepath)
        
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({
            'success': True,
            'ingredients': ingredients
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-recipe', methods=['POST'])
def generate_recipe():
    data = request.get_json()
    
    if not data or 'ingredients' not in data:
        return jsonify({'error': 'No ingredients provided'}), 400
    
    ingredients = data['ingredients']
    
    if not ingredients or len(ingredients) == 0:
        return jsonify({'error': 'Ingredients list is empty'}), 400
    
    try:
        generator = get_recipe_generator()
        recipe = generator.generate(ingredients)
        
        return jsonify({
            'success': True,
            'recipe': recipe
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_and_generate():

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        # Save uploaded file
        filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Detect ingredients
        detector = get_food_detector()
        ingredients = detector.detect(filepath)
        
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)
        
        if not ingredients:
            return jsonify({
                'success': False,
                'error': 'No food ingredients detected in the image'
            })
        
        # Generate recipe
        generator = get_recipe_generator()
        recipe = generator.generate([ing['name'] for ing in ingredients])
        
        return jsonify({
            'success': True,
            'ingredients': ingredients,
            'recipe': recipe
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting Food Recipe API...")
    print("Using CLIP model for food ingredient detection (111 ingredients)")
    app.run(debug=True, host='0.0.0.0', port=5000)
