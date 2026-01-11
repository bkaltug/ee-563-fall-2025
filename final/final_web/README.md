# AI Recipe Generator

A web application that identifies food ingredients from images using MMDetection and generates recipes using HuggingFace LLM models.

![Demo](demo-screenshot.png)

## Features

- 📷 **Image Upload**: Drag & drop or click to upload food images
- 🔍 **Ingredient Detection**: Uses MMDetection for accurate food ingredient recognition
- 🤖 **AI Recipe Generation**: Leverages HuggingFace models to create recipes
- ✏️ **Edit Ingredients**: Modify detected ingredients before generating recipes
- 🖨️ **Print Recipes**: Print-friendly recipe output
- 📱 **Responsive Design**: Works on desktop and mobile devices

## Project Structure

```
final_web/
├── backend/
│   ├── app.py                 # Flask API server
│   ├── requirements.txt       # Python dependencies
│   ├── services/
│   │   ├── __init__.py
│   │   ├── image_recognition.py   # MMDetection food detection
│   │   └── recipe_generator.py    # HuggingFace LLM integration
│   ├── models/                # Store model weights here
│   └── uploads/               # Temporary image uploads
├── frontend/
│   ├── index.html            # Main HTML page
│   ├── styles.css            # CSS styling
│   └── script.js             # JavaScript functionality
└── README.md
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Node.js (optional, for frontend development)
- CUDA-capable GPU (recommended for faster inference)

### Step 1: Clone and Navigate

```bash
cd final_web
```

### Step 2: Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### Step 3: Install PyTorch

Visit [PyTorch.org](https://pytorch.org/get-started/locally/) and select your configuration.

```bash
# Example for CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Example for CPU only:
pip install torch torchvision torchaudio
```

### Step 4: Install MMDetection

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
```

### Step 5: Install Other Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Step 6: Download Model Weights

For MMDetection, download a pre-trained model:

```bash
# Create models directory
mkdir -p models

# Download Faster R-CNN config and weights
mim download mmdet --config faster_rcnn_r50_fpn_1x_coco --dest models/
```

Or download directly:
- Config: [faster_rcnn_r50_fpn_1x_coco.py](https://github.com/open-mmlab/mmdetection/tree/main/configs/faster_rcnn)
- Weights: [faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth)

### Step 7: Set Up HuggingFace (Optional)

For better recipe generation, get a HuggingFace API token:

1. Create account at [huggingface.co](https://huggingface.co)
2. Generate API token in Settings
3. Set environment variable:

```bash
# Windows
set HUGGINGFACE_API_TOKEN=your_token_here

# Linux/Mac
export HUGGINGFACE_API_TOKEN=your_token_here
```

## Running the Application

### Start the Backend Server

```bash
cd backend
python app.py
```

The API will be available at `http://localhost:5000`

### Access the Frontend

Open a web browser and navigate to:
```
http://localhost:5000
```

Or serve the frontend separately:
```bash
cd frontend
python -m http.server 8080
```
Then visit `http://localhost:8080`

## API Endpoints

### Health Check
```
GET /api/health
```

### Detect Ingredients
```
POST /api/detect
Content-Type: multipart/form-data
Body: image (file)
```

Response:
```json
{
  "success": true,
  "ingredients": [
    {"name": "tomato", "confidence": 0.95},
    {"name": "egg", "confidence": 0.88}
  ]
}
```

### Generate Recipe
```
POST /api/generate-recipe
Content-Type: application/json
Body: {"ingredients": ["tomato", "egg", "onion"]}
```

Response:
```json
{
  "success": true,
  "recipe": {
    "name": "Tomato Egg Stir-Fry",
    "ingredients": ["4 eggs", "2 tomatoes", ...],
    "instructions": ["Beat eggs...", ...],
    "cooking_time": "15 minutes",
    "servings": "2-3 servings",
    "tips": "..."
  }
}
```

### Combined Analysis
```
POST /api/analyze
Content-Type: multipart/form-data
Body: image (file)
```

## Configuration

### Using Custom Models

#### Custom Food Detection Model

1. Train your model on a food dataset (e.g., Food-101, UECFOOD)
2. Place config and checkpoint in `backend/models/`
3. Update paths in `image_recognition.py`

#### Custom LLM for Recipes

Update `recipe_generator.py`:

```python
generator = RecipeGenerator(
    model_name='your-model-name',
    use_api=True,  # or False for local
    api_token='your-token'
)
```

### Supported HuggingFace Models

- `gpt2`, `gpt2-medium`, `distilgpt2`
- `bigscience/bloom-560m`
- `google/flan-t5-base`
- `flax-community/t5-recipe-generation`
- `mistralai/Mistral-7B-Instruct-v0.1`
- `meta-llama/Llama-2-7b-chat-hf`

## Troubleshooting

### MMDetection Not Working

The app includes a mock detection fallback that analyzes image colors to suggest ingredients. This allows testing without full MMDetection setup.

### CUDA Out of Memory

- Use smaller models
- Enable 8-bit quantization with `bitsandbytes`
- Use CPU inference (slower but works)

### HuggingFace Rate Limits

- Use local models instead of API
- Get a Pro account for higher limits

## Development

### Frontend Development

The frontend is vanilla HTML/CSS/JS. Simply edit the files in `frontend/` and refresh.

### Backend Development

```bash
cd backend
python app.py  # Runs in debug mode
```

### Adding New Food Classes

Edit `FOOD_CLASSES` in `image_recognition.py` and `coco_to_food` mapping.

## Future Improvements

- [ ] Fine-tune MMDetection on food-specific datasets
- [ ] Add user authentication
- [ ] Save favorite recipes
- [ ] Nutrition information
- [ ] Multiple recipe suggestions
- [ ] Dietary restriction filters
- [ ] Shopping list generation

## License

MIT License

## Acknowledgments

- [MMDetection](https://github.com/open-mmlab/mmdetection) for object detection
- [HuggingFace](https://huggingface.co) for LLM models
- [Flask](https://flask.palletsprojects.com/) for the backend framework
