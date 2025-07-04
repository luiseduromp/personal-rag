# Personal RAG Application

This is a Retrieval-Augmented Generation (RAG) system that serves as a personal AI assistant, capable of answering questions about my professional background, skills, and experiences. It uses modern LLMs to process and generate responses based on documents that I have provided to the system.

## 🌟 Features

- **Document Retrieval**: Efficiently searches through uploaded documents to find relevant information
- **Conversational AI**: Maintains context across multiple questions for natural conversations
- **Secure Access**: Token-based authentication to protect personal information
- **Document Management**: Easily add new documents via URL for the AI to learn from
- **Context-Aware Responses**: Provides accurate, relevant answers based on the provided documents
- **Source Attribution**: Includes references to the source documents for transparency

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python)
- **Vector Database**: ChromaDB
- **Language Model**: OpenAI API
- **Orchestrator**: LangChain
- **Embeddings**: OpenAI Embeddings
- **Authentication**: JWT
- **Package Management**: Poetry

## 🚀 API Endpoints

### Authentication

- `POST /token` - Obtain an access token for API authentication

### Main Endpoints

- `GET /` - Basic health check endpoint
- `POST /generate` - Generate an answer to a question using the RAG pipeline
- `POST /upload` - Add a new document to the knowledge base via URL``

## 🚀 Getting Started

### Prerequisites

- Python 3.12 or higher
- Poetry (for dependency management)
- OpenAI API key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/personal-rag.git
   cd personal-rag
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Set up environment variables:
   Create a `.env` file in the project root with the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key
   JWT_SECRET_KEY=your_jwt_secret_key
   JWT_ALGORITHM=HS256
   JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
   ```

4. Initialize the knowledge base:
   - Place your documents in the `app/docs/` directory
   - The application will automatically process these documents on startup

### Running the Application

1. Start the FastAPI server:
   ```bash
   poetry run uvicorn app.main:app --reload
   ```

2. The API will be available at `http://localhost:8000`

3. Access the interactive API documentation at `http://localhost:8000/docs`

## 🐳 Docker Support

You can also run the application using Docker:

```bash
# Build the Docker image
docker build -t personal-rag .

# Run the container
docker run -p 8000:8000 --env-file .env personal-rag
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

Luis Romero - [@luiseduromp](https://github.com/luiseduromp) - luiseduromp@gmail.com

Project Link: [https://github.com/your-username/personal-rag](https://github.com/your-username/personal-rag)
