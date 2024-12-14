# SmartChef Backend 🚀

The backend infrastructure for SmartChef, an AI-powered recipe recommendation and cooking assistance platform. This system combines BERT and TF-IDF models with Llama 3.2 for sophisticated recipe understanding and intelligent chat interactions.

## 🌟 Features

### AI Models

- **Dual Model Architecture**
  - BERT Model for semantic recipe understanding
  - TF-IDF Model for efficient text-based search
  - Model switching capability for optimized performance
- **Recipe Enhancement**
  - Natural language processing for recipe steps
  - Ingredient parsing and normalization
  - Equipment detection
  - Timing extraction
- **Image Service**
  - Automatic recipe image fetching
  - Caching system with 7-day retention
  - Parallel image processing

### Chat System

- **Llama 3.2 Integration**
  - Uses the 500000+ recipe dataset for recommendations
  - Context-aware recipe suggestions
  - Health and dietary consideration
  - Personalized cooking guidance
- **WebSocket Implementation**
  - Real-time bidirectional communication
  - Automatic reconnection handling
  - Message queuing
- **Chat Persistence**
  - SQLite database storage
  - Message history management
  - Session handling

## 🚀 Setup and Installation

> [!WARNING]
> This is an academic major project created for educational purposes. The project does not implement comprehensive security measures and may contain vulnerabilities. I take no responsibility for any issues, data breaches, or problems that may occur from using this codebase. If you choose to use or modify this code, you do so entirely at your own risk. This is not intended for production use without significant security enhancements and proper testing. By using this project, you acknowledge and accept these terms.

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Ollama with Llama 3.2 installed

## 📥 Required Data Files

Due to GitHub's file size limitations, the data and cache folders are hosted separately. Download them from:
[📁Google Drive](https://drive.google.com/drive/folders/1e-5LC-UOkEEXBdP15MoKk9-TQpKggk3H?usp=drive_link)

### Setup Instructions for Data Files:

1. Download both `data` and `cache` folders from the Google Drive link
2. Place them in the root directory of the project
3. Ensure the following structure:
   ```
   SmartChef-backend/
   ├── data/
   │   ├── recipes.csv              # Recipe dataset
   │   └── processed_recipes.pkl    # Preprocessed data
   └── cache/
       ├── bert/                    # BERT model cache
       ├── tfidf/                   # TF-IDF model cache
       └── images/                  # Image cache
   ```

> [!IMPORTANT]
> The system requires these files for proper functioning. Make sure to download and place them correctly before starting the server.

### Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/Debojit-mitra/SmartChef-backend.git
   cd SmartChef-backend
   ```

2. Create and activate virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Install spaCy model:

   ```bash
   python -m spacy download en_core_web_sm
   ```

5. To run:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

## 🔌 API Endpoints

### Recipe Endpoints

- `POST /api/v1/recommend/ingredients`

  - **Request Body:**
    ```json
    {
      "ingredients": ["ingredient1", "ingredient2"],
      "model_type": "bert",
      "top_n": 5
    }
    ```
  - **Fields:**
    - `ingredients` (array): List of ingredients
    - `model_type` (string): Model to use ("bert" or "tfidf")
    - `top_n` (integer): Number of results to return

- `GET /api/v1/recipe/{recipe_name}?model_type={model_type}&top_n={no_of_results}`
  - **Parameters:**
    - `recipe_name` (path): Name of the recipe to search
    - `model_type` (query): Model to use for search (default: "bert")
    - Options: "bert", "tfidf"
    - `top_n` (query): Number of results to return (default: 5)

### System Endpoints

- `GET /health`
  - System health check
- `POST /api/v1/cache/clear`
  - Clear system cache
- `GET /api/v1/cache/stats`
  - Get cache statistics
- `POST /api/v1/models/refresh`
  - Refresh AI models

### WebSocket

- `WS /ws/chat`
  - Real-time chat connection

## 🚀 Running the Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/yourfeature`)
3. Commit your changes (`git commit -m 'Add yourfeature'`)
4. Push to the branch (`git push origin feature/yourfeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Made with ❤️ by Debojit
