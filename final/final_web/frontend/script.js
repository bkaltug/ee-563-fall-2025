/**
 * AI Recipe Generator - Main JavaScript
 * Handles image upload, API communication, and UI interactions
 */

// Configuration
const API_BASE_URL = 'http://localhost:5000/api';

// State
let currentIngredients = [];
let selectedFile = null;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const uploadPlaceholder = document.getElementById('uploadPlaceholder');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const imageInput = document.getElementById('imageInput');
const removeBtn = document.getElementById('removeBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingText = document.getElementById('loadingText');
const resultsSection = document.getElementById('resultsSection');
const ingredientsList = document.getElementById('ingredientsList');
const editIngredientsBtn = document.getElementById('editIngredientsBtn');
const generateRecipeBtn = document.getElementById('generateRecipeBtn');
const editModal = document.getElementById('editModal');
const editIngredientsContainer = document.getElementById('editIngredientsContainer');
const newIngredientInput = document.getElementById('newIngredientInput');
const addIngredientBtn = document.getElementById('addIngredientBtn');
const cancelEditBtn = document.getElementById('cancelEditBtn');
const saveIngredientsBtn = document.getElementById('saveIngredientsBtn');
const recipePanel = document.getElementById('recipePanel');
const recipeContent = document.getElementById('recipeContent');
const printRecipeBtn = document.getElementById('printRecipeBtn');
const newRecipeBtn = document.getElementById('newRecipeBtn');
const startOverBtn = document.getElementById('startOverBtn');
const toast = document.getElementById('toast');
const toastMessage = document.getElementById('toastMessage');

// Initialize Event Listeners
function init() {
    // Upload area click
    uploadArea.addEventListener('click', () => {
        if (!selectedFile) {
            imageInput.click();
        }
    });

    // File input change
    imageInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);

    // Remove image
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        clearImage();
    });

    // Analyze button
    analyzeBtn.addEventListener('click', analyzeImage);

    // Edit ingredients
    editIngredientsBtn.addEventListener('click', openEditModal);
    cancelEditBtn.addEventListener('click', closeEditModal);
    saveIngredientsBtn.addEventListener('click', saveIngredients);
    addIngredientBtn.addEventListener('click', addNewIngredient);
    newIngredientInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') addNewIngredient();
    });

    // Generate recipe
    generateRecipeBtn.addEventListener('click', generateRecipe);

    // Recipe actions
    printRecipeBtn.addEventListener('click', () => window.print());
    newRecipeBtn.addEventListener('click', generateRecipe);
    startOverBtn.addEventListener('click', resetApp);

    // Close modal on outside click
    editModal.addEventListener('click', (e) => {
        if (e.target === editModal) closeEditModal();
    });
}

// File Handling Functions
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        processFile(file);
    }
}

function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const file = e.dataTransfer.files[0];
    if (file) {
        processFile(file);
    }
}

function processFile(file) {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/jpg'];
    if (!validTypes.includes(file.type)) {
        showToast('Please upload a valid image file (JPG, PNG, or WEBP)', 'error');
        return;
    }

    // Validate file size (16MB)
    if (file.size > 16 * 1024 * 1024) {
        showToast('File size must be less than 16MB', 'error');
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        uploadPlaceholder.style.display = 'none';
        imagePreview.style.display = 'block';
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

function clearImage() {
    selectedFile = null;
    imageInput.value = '';
    previewImg.src = '';
    uploadPlaceholder.style.display = 'block';
    imagePreview.style.display = 'none';
    analyzeBtn.disabled = true;
}

// API Functions
async function analyzeImage() {
    if (!selectedFile) {
        showToast('Please select an image first', 'error');
        return;
    }

    showLoading('Analyzing your image...');

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
        const response = await fetch(`${API_BASE_URL}/detect`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to analyze image');
        }

        if (data.success && data.ingredients) {
            currentIngredients = data.ingredients;
            displayIngredients(currentIngredients);
            resultsSection.style.display = 'grid';
            recipePanel.style.display = 'none';
            showToast(`Found ${currentIngredients.length} ingredients!`, 'success');
        } else {
            showToast('No ingredients detected. Try another image.', 'error');
        }
    } catch (error) {
        console.error('Error analyzing image:', error);
        showToast(error.message || 'Failed to analyze image', 'error');
    } finally {
        hideLoading();
    }
}

async function generateRecipe() {
    if (currentIngredients.length === 0) {
        showToast('No ingredients to generate recipe from', 'error');
        return;
    }

    showLoading('Generating your recipe...');

    try {
        const ingredientNames = currentIngredients.map(ing => 
            typeof ing === 'string' ? ing : ing.name
        );

        const response = await fetch(`${API_BASE_URL}/generate-recipe`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ ingredients: ingredientNames })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to generate recipe');
        }

        if (data.success && data.recipe) {
            displayRecipe(data.recipe);
            recipePanel.style.display = 'block';
            recipePanel.scrollIntoView({ behavior: 'smooth' });
            showToast('Recipe generated successfully!', 'success');
        } else {
            showToast('Failed to generate recipe. Please try again.', 'error');
        }
    } catch (error) {
        console.error('Error generating recipe:', error);
        showToast(error.message || 'Failed to generate recipe', 'error');
    } finally {
        hideLoading();
    }
}

// Display Functions
function displayIngredients(ingredients) {
    ingredientsList.innerHTML = '';
    
    ingredients.forEach(ing => {
        const name = typeof ing === 'string' ? ing : ing.name;
        const confidence = typeof ing === 'object' && ing.confidence 
            ? Math.round(ing.confidence * 100) 
            : null;
        
        const tag = document.createElement('div');
        tag.className = 'ingredient-tag';
        tag.innerHTML = `
            <span>${capitalizeFirst(name)}</span>
            ${confidence ? `<span class="ingredient-confidence">${confidence}%</span>` : ''}
        `;
        ingredientsList.appendChild(tag);
    });
}

function displayRecipe(recipe) {
    const html = `
        <h3 class="recipe-name">${recipe.name || 'Delicious Recipe'}</h3>
        
        <div class="recipe-meta">
            ${recipe.cooking_time ? `
                <div class="meta-item">
                    <span class="meta-icon">⏱️</span>
                    <div>
                        <div class="meta-label">Cooking Time</div>
                        <div class="meta-value">${recipe.cooking_time}</div>
                    </div>
                </div>
            ` : ''}
            ${recipe.servings ? `
                <div class="meta-item">
                    <span class="meta-icon">👥</span>
                    <div>
                        <div class="meta-label">Servings</div>
                        <div class="meta-value">${recipe.servings}</div>
                    </div>
                </div>
            ` : ''}
        </div>

        <div class="recipe-section">
            <h4>🥘 Ingredients</h4>
            <ul class="recipe-ingredients-list">
                ${(recipe.ingredients || []).map(ing => `<li>${ing}</li>`).join('')}
            </ul>
        </div>

        <div class="recipe-section">
            <h4>📝 Instructions</h4>
            <ol class="recipe-instructions">
                ${(recipe.instructions || []).map(step => `<li>${step}</li>`).join('')}
            </ol>
        </div>

        ${recipe.tips ? `
            <div class="recipe-section">
                <h4>💡 Tips & Variations</h4>
                <div class="recipe-tips">
                    <p>${recipe.tips}</p>
                </div>
            </div>
        ` : ''}
    `;

    recipeContent.innerHTML = html;
}

// Edit Modal Functions
function openEditModal() {
    editIngredientsContainer.innerHTML = '';
    
    currentIngredients.forEach((ing, index) => {
        const name = typeof ing === 'string' ? ing : ing.name;
        const div = document.createElement('div');
        div.className = 'editable-ingredient';
        div.innerHTML = `
            <span>${capitalizeFirst(name)}</span>
            <button onclick="removeIngredient(${index})">✕</button>
        `;
        editIngredientsContainer.appendChild(div);
    });
    
    editModal.style.display = 'flex';
}

function closeEditModal() {
    editModal.style.display = 'none';
    newIngredientInput.value = '';
}

function addNewIngredient() {
    const value = newIngredientInput.value.trim();
    if (value) {
        currentIngredients.push({ name: value, confidence: 1.0 });
        openEditModal(); // Refresh modal
        newIngredientInput.value = '';
    }
}

// Global function for inline onclick
window.removeIngredient = function(index) {
    currentIngredients.splice(index, 1);
    openEditModal(); // Refresh modal
};

function saveIngredients() {
    displayIngredients(currentIngredients);
    closeEditModal();
    showToast('Ingredients updated!', 'success');
}

// Utility Functions
function showLoading(message) {
    loadingText.textContent = message;
    loadingOverlay.classList.add('active');
}

function hideLoading() {
    loadingOverlay.classList.remove('active');
}

function showToast(message, type = 'info') {
    toastMessage.textContent = message;
    toast.className = 'toast show';
    if (type) {
        toast.classList.add(type);
    }
    
    setTimeout(() => {
        toast.className = 'toast';
    }, 3000);
}

function capitalizeFirst(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

function resetApp() {
    clearImage();
    currentIngredients = [];
    resultsSection.style.display = 'none';
    recipePanel.style.display = 'none';
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', init);
