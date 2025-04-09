# AI Assistant with Ollama and Streamlit

A powerful AI assistant that combines document-based question answering with casual conversation capabilities using Ollama models and Streamlit.

## Features

- **Document Processing**: Upload and process PDF, TXT, and DOCX files
- **Document Q&A**: Ask questions about uploaded documents
- **Casual Conversation**: Engage in general chat when not discussing documents
- **Model Selection**: Choose from available Ollama models
- **Vector Search**: Efficient document search using FAISS
- **Modern UI**: Clean and intuitive Streamlit interface

## Prerequisites

- Python 3.10 or higher
- Ollama installed and running locally
- At least one Ollama model downloaded (e.g., gemma3:4b)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sushil79g/AI_ASSISTANT.git
cd AI_ASSISTANT
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start Ollama service (if not already running):
```bash
ollama serve
```

2. Run the Streamlit app:
```bash
streamlit run src/app.py
```

3. Access the app at `http://localhost:8501`

## Features in Detail

### Document Processing
- Supports PDF, TXT, and DOCX files
- Extracts text content for question answering
- Splits documents into manageable chunks
- Creates vector embeddings for efficient search

### Question Answering
- Answers questions based ONLY on uploaded documents
- Provides clear messages when answers aren't in the document
- Uses semantic search to find relevant information
- Combines multiple relevant chunks for better context

### Casual Conversation
- Engages in general chat when not discussing documents
- Maintains context within conversations
- Provides helpful and friendly responses
- Clearly indicates when switching between document Q&A and chat

### Model Management
- Automatically detects available Ollama models
- Allows switching between different models
- Provides refresh button to update model list
- Handles model availability gracefully

## CI/CD Pipeline

The project includes a GitHub Actions workflow that:
1. Runs tests on the code
2. Builds a Docker image
3. Tests the Docker image
4. Reports test coverage

## Project Structure

```
.
├── src/
│   ├── app.py                 # Main Streamlit application
│   ├── models/
│   │   └── ollama_client.py   # Ollama API client
│   ├── document_processor/
│   │   └── processor.py       # Document processing logic
│   └── utils/
│       └── qa_handler.py      # Question answering handler
├── tests/                     # Test files
├── requirements.txt           # Python dependencies
├── Dockerfile                # Docker configuration
└── README.md                 # Project documentation
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
