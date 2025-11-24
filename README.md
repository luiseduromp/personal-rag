7# Personal RAG Application

This is a multilingual Retrieval-Augmented Generation (RAG) system that serves as a personal AI assistant, capable of answering
questions about my professional background, skills, and experience, in English and Spanish 🎉. It uses an LLM to process and 
generate responses based on documents that I have provided to the system.

🔗 Try the application in [my portfolio](https://luiseduromp.com) 

## 🌟 Features

- **Document Retrieval**: Searches through provided documents to find relevant information.
- **Conversational AI**: Maintains context across multiple questions for natural conversations.
- **Multilingual Response**: Supports conversations in English and Spanish using a dual RAG pipeline
- **Secure Access**: Token-based authentication to prevent API abuse.
- **Document Management**: Read documents from disk or from a remote S3 bucket. Add documents easily.

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python)
- **Vector Database**: ChromaDB
- **Language Model**: GPT-4o-mini from OpenAI
- **Orchestrator**: LangChain
- **Embeddings**: OpenAI Embeddings
- **Authentication**: JWT
- **Optional remote directory**: AWS S3, CloudFront, Lambda

## 🚀 API Endpoints

### Authentication

- `POST /token` - Obtain an access token for API authentication

### Main Endpoints

- `GET /` - Basic health check endpoint
- `POST /generate` - Generate an answer to a question using the RAG pipeline
- `POST /upload` - Add a new document to the knowledge base via URL

***Note.*** The `/upload` endpoint currently supports only documents available via URL, such as a CDN. 


## 🧪 Trying the app
If you want to setup this application for your own use, follow the steps below.
Keep in mind that this does not cover the prerequisite installation of Python, Poetry, the OpenAI API key creation or the 
remote directory setup (S3 bucket, CloudFront, Lambda setups)

### Prerequisites

- Python 3.12 or higher
- Poetry (for dependency management)
- OpenAI API key

### Document Loading
The documents are loaded from two sources: the `app/docs` directory in the disk, and a remote directory in an AWS S3 bucket.
The remote directory is optional. The app has been designed assuming the remote (bucket) directory is private, so the list 
of available files are read using a Lambda function, and the files are served through a CloudFront CDN.
Therefore, for the remote directory, there is an aditional setup to be done which is not covered here, and it requires the 
URL to trigger the Lambda function and the CDN URL.

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/luiseduromp/personal-rag.git
   cd personal-rag
   ```

2. Install dependencies using uv:
   ```bash
   uv sync
   ```

3. Set up environment variables:
   Create a `.env` file in the project root as the .env.example
   
   The `allowed_origins` is a comma separated list of origins that are allowed to access the API. It is used as protection to 
   prevent API abuse. Leave as `*` to allow all origins.  

   The `secret key` can be generated using a random string generator.

   The `username` and `hashed` password are used for authentication. This is a simple authentication method to prevent abuse of
   the API. You can create a hashed password using bcrypt. Here is a simple python script to do it:
   ```python
   import bcrypt
   password = b"your_password"
   hashed = bcrypt.hashpw(password, bcrypt.gensalt())
   print(hashed)
   ```

4. Initialize the knowledge base:
   - Place your documents in the `app/docs/` directory. The supported documents are PDF, TXT and MD.
   - The application will automatically process these documents on startup.

5. **Optional.** Add a remote document directory:
   - Place your documents in your AWS S3 bucket in a `/docs` folder and create a Lambda function [Lambda function](/lambda/list_files.py)
   to read the documents.
   - Add the API Gateway URL and the CloudFront CDN URL in your environment variables.


### Running the Application

1. Start the FastAPI server:
   ```bash
   uv run uvicorn app.main:app --reload
   ```

2. The API will be available at `http://localhost:8000`

3. Use Postman or any other API client to test the application. Remember to get an access token before making requests.


## 🐳 Docker Support

You can also run the application using Docker:

```bash
docker build -t personal-rag .

docker run -p 8000:8000 --env-file .env personal-rag
```

## 🤝 Contributing

Feedback and contributions are welcome! Please feel free to submit a Pull Request.
