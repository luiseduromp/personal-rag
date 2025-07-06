# Personal RAG Application

This is a Retrieval-Augmented Generation (RAG) system that serves as a personal AI assistant, capable of answering questions about my professional background, skills, and experience. It uses an LLM to process and generate responses based on documents that I have provided to the system.

## 🌟 Features

- **Document Retrieval**: Searches through provided documents to find relevant information.
- **Conversational AI**: Maintains context across multiple questions for natural conversations.
- **Secure Access**: Token-based authentication to prevent API abuse.
- **Document Management**: Easily add new documents. Right now it supports only adding documents from a URL.

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python)
- **Vector Database**: ChromaDB
- **Language Model**: GPT-4o-mini from OpenAI
- **Orchestrator**: LangChain
- **Embeddings**: OpenAI Embeddings
- **Authentication**: JWT

## 🚀 API Endpoints

### Authentication

- `POST /token` - Obtain an access token for API authentication

### Main Endpoints

- `GET /` - Basic health check endpoint
- `POST /generate` - Generate an answer to a question using the RAG pipeline
- `POST /upload` - Add a new document to the knowledge base via URL``

## 🧪 Trying the app
If you want to setup this application for your own use, follow the steps below. Keep in mind that this does not cover the prerequisite installation of Python, Poetry, or the OpenAI API key creation.

### Prerequisites

- Python 3.12 or higher
- Poetry (for dependency management)
- OpenAI API key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/luiseduromp/personal-rag.git
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
   SECRET_KEY=your_secret_key
   ALGORITHM=HS256
   USERNAME=your_username
   HASHED=your_hashed_password
   ALLOWED_ORIGINS=your_allowed_origins
   ```
   The `allowed_origins` is a comma separated list of origins that are allowed to access the API. It is used as protection to prevent API abuse. Leave as `*` to allow all origins.  

   The `secret key` can be generated using a random string generator.

   The `username` and `hashed` password are used for authentication. This is a simple authentication method to prevent abuse of the API. 
   You can create a hashed password using bcrypt. Here is a simple python script to do it:
   ```python
   import bcrypt
   password = b"your_password"
   hashed = bcrypt.hashpw(password, bcrypt.gensalt())
   print(hashed)
   ```

4. Initialize the knowledge base:
   - Place your documents in the `app/docs/` directory. You may need to create the directory. The supported documents are PDF, TXT and MD.
   - The application will automatically process these documents on startup

### Running the Application

1. Start the FastAPI server:
   ```bash
   poetry run uvicorn app.main:app --reload
   ```

2. The API will be available at `http://localhost:8000`

3. Use Postman or any other API client to test the application.

## 🐳 Docker Support

You can also run the application using Docker:

```bash
docker build -t personal-rag .

docker run -p 8000:8000 --env-file .env personal-rag
```

## 🤝 Contributing

Feedback and contributions are welcome! Please feel free to submit a Pull Request.
