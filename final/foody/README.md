# Foody 🍳

A Flutter mobile app that helps you discover recipes based on the ingredients you have! Simply take a photo of your ingredients, and the app will recognize them using Google ML Kit and generate a delicious recipe using Google Gemini AI.

## Features

- 📸 **Capture Ingredients**: Take a photo or select from gallery
- 🔍 **AI-Powered Recognition**: Uses Google ML Kit for image labeling to detect ingredients
- 🤖 **Smart Recipe Generation**: Leverages Google Gemini AI to create personalized recipes
- 📱 **Cross-Platform**: Works on Android and iOS

## Setup

### Prerequisites

1. Flutter SDK (3.9.2 or higher)
2. A Google Gemini API key

### Getting Your Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create or sign in to your Google account
3. Generate a new API key
4. Copy the API key

### Configuration

1. Open `lib/services/gemini_service.dart`
2. Replace `YOUR_GEMINI_API_KEY` with your actual API key:
   ```dart
   static const String _apiKey = 'your-actual-api-key-here';
   ```

### Installation

```bash
# Get dependencies
flutter pub get

# Run the app
flutter run
```

## Project Structure

```
lib/
├── main.dart                 # App entry point
├── screens/
│   └── home_screen.dart      # Main UI screen
└── services/
    ├── gemini_service.dart   # Google Gemini API integration
    └── image_labeling_service.dart  # ML Kit image labeling
```

## How It Works

1. **Capture**: User takes a photo of their ingredients
2. **Analyze**: Google ML Kit processes the image and identifies ingredients
3. **Generate**: Detected ingredients are sent to Google Gemini API
4. **Display**: A customized recipe is displayed to the user

## Permissions

### Android
- Camera
- Read/Write External Storage
- Internet

### iOS
- Camera Usage
- Photo Library Usage

## Technologies Used

- **Flutter**: Cross-platform mobile development
- **Google ML Kit**: On-device image labeling
- **Google Gemini AI**: Recipe generation
- **image_picker**: Camera and gallery integration
- **permission_handler**: Runtime permissions

## License

This project is for educational purposes (EE563 Class Project).
