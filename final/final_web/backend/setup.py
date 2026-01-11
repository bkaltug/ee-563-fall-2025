#!/usr/bin/env python
"""
Setup script for downloading required model weights
Run this after installing dependencies
"""

import os
import subprocess
import sys

def create_directories():
    """Create necessary directories"""
    dirs = ['models', 'uploads']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"✓ Created directory: {d}")

def download_mmdetection_model():
    """Download MMDetection pre-trained model"""
    print("\n📥 Downloading MMDetection model...")
    
    try:
        # Check if mim is installed
        import mim
        
        # Download config and checkpoint
        subprocess.run([
            sys.executable, '-m', 'mim', 'download', 'mmdet',
            '--config', 'faster_rcnn_r50_fpn_1x_coco',
            '--dest', 'models/'
        ], check=True)
        
        print("✓ MMDetection model downloaded successfully!")
        return True
        
    except ImportError:
        print("⚠ MIM not installed. Installing...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-U', 'openmim'], check=True)
        return download_mmdetection_model()
        
    except Exception as e:
        print(f"✗ Failed to download MMDetection model: {e}")
        print("  You can manually download from:")
        print("  https://github.com/open-mmlab/mmdetection/tree/main/configs/faster_rcnn")
        return False

def download_huggingface_model():
    """Download HuggingFace model for recipe generation"""
    print("\n📥 Downloading HuggingFace model...")
    
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        
        model_name = "google/flan-t5-base"
        
        print(f"  Downloading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Save locally
        save_path = os.path.join('models', 'flan-t5-base')
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        
        print(f"✓ HuggingFace model saved to: {save_path}")
        return True
        
    except ImportError:
        print("⚠ Transformers not installed.")
        print("  Install with: pip install transformers")
        print("  Skipping model download - the app will use mock recipes.")
        return False
        
    except Exception as e:
        print(f"✗ Failed to download HuggingFace model: {e}")
        print("  The app will download the model on first use or use mock recipes.")
        return False

def verify_installation():
    """Verify that all components are properly installed"""
    print("\n🔍 Verifying installation...")
    
    checks = []
    
    # Check Flask
    try:
        import flask
        checks.append(("Flask", True, flask.__version__))
    except ImportError:
        checks.append(("Flask", False, "Not installed"))
    
    # Check PyTorch
    try:
        import torch
        cuda_status = "CUDA available" if torch.cuda.is_available() else "CPU only"
        checks.append(("PyTorch", True, f"{torch.__version__} ({cuda_status})"))
    except ImportError:
        checks.append(("PyTorch", False, "Not installed"))
    
    # Check MMDetection
    try:
        import mmdet
        checks.append(("MMDetection", True, mmdet.__version__))
    except ImportError:
        checks.append(("MMDetection", False, "Not installed"))
    
    # Check Transformers
    try:
        import transformers
        checks.append(("Transformers", True, transformers.__version__))
    except ImportError:
        checks.append(("Transformers", False, "Not installed"))
    
    # Check PIL
    try:
        from PIL import Image
        checks.append(("Pillow", True, "Installed"))
    except ImportError:
        checks.append(("Pillow", False, "Not installed"))
    
    # Print results
    print("\n" + "="*50)
    print("Installation Status")
    print("="*50)
    
    all_ok = True
    for name, status, version in checks:
        icon = "✓" if status else "✗"
        print(f"{icon} {name}: {version}")
        if not status:
            all_ok = False
    
    print("="*50)
    
    if all_ok:
        print("\n🎉 All dependencies installed successfully!")
    else:
        print("\n⚠ Some dependencies are missing. Please install them.")
    
    return all_ok

def main():
    print("="*50)
    print("AI Recipe Generator - Setup Script")
    print("="*50)
    
    # Create directories
    create_directories()
    
    # Download models
    download_mmdetection_model()
    download_huggingface_model()
    
    # Verify installation
    verify_installation()
    
    print("\n📖 Next steps:")
    print("1. Copy .env.example to .env and configure")
    print("2. Run: python app.py")
    print("3. Open http://localhost:5000 in your browser")
    print("\nHappy cooking! 🍳")

if __name__ == '__main__':
    main()
