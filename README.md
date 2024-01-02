# LLM-Based Power Virtual Agent and Power Automate Chatbot

This project leverages the power of OpenAI's GPT models using Langchain to provide a robust chatbot for Power Virtual Agent and Power Automate platforms. Our solution enriches backend operations with advanced NLP capabilities to process user input effectively and generate meaningful responses.

## Features

- **LLM Backend**: Uses Azure deployment of OpenAI's GPT models.
- **Data Preprocessing**: Advanced data preprocessing module to optimize user input for GPT model consumption.
- **Prompt Engineering**: Utilizes prompt templates to guide the GPT models and produce tailored responses suitable for chatbot interactions.
- **Integration with Power Platforms**: Designed to be seamlessly integrated with Microsoft's Power Virtual Agent and Power Automate platforms.

## Prerequisites

1. Python 3.11 or higher.
2. OpenAI Azure Deployment.
3. Microsoft's Power Virtual Agent and Power Automate platforms account.

## Setup

1. **Clone the Repository**:
    ```bash
    git clone [repo_link]
    cd [repo_directory]
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Environment Variables**:
    Copy the `.env.example` to `.env` and update the variables including the OpenAI API Key, API base URL, etc.

4. **Run the Backend**:
    ```bash
    python api.py
    ```

## Usage

1. Start the Streamlit UI:
    ```bash
    python app.py
    ```

2. Navigate to the provided URL in your browser. Input the desired genre, characters, and news text. Click on "Begin!" to get the novel based on your inputs.

## Contributing

We welcome contributions! Please create an issue to discuss the proposed changes or submit a pull request.

## License

[MIT License](LICENSE)
