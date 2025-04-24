# BERT-based Intent Classification Chatbot

This is a simple chatbot application that uses **BERT (Bidirectional Encoder Representations from Transformers)** to classify user input into predefined intents and generate appropriate responses. It uses `transformers` from Hugging Face and is trained on custom intent data from a JSON file.

## Features

- Fine-tunes BERT for intent classification using your own labeled data.
- Predicts the intent of the user input and returns a suitable response.
- CLI-based chatbot interaction.
- Easy to extend with new intents and patterns.

## Project Structure

bert_chatbot/ ├── intents.json # Dataset with patterns, tags, and responses 
              ├── bert_chatbot.py # Main script with training and chatbot logic 
              └── README.md # Project description and setup guide

## Requirements

- Make sure you have Python 3.7+ installed.
- Install required libraries using pip:

```bash
pip install transformers datasets scikit-learn torch
```

## Dataset Format (intents.json)
The data is stored as a list of intents with:
- tag: category of the user query
- patterns: example inputs for that tag
- responses: responses for that intent

Example:

```json
[
  {
    "tag": "greeting",
    "patterns": ["Hi", "Hello", "Hey", "How are you?"],
    "responses": ["Hello!", "Hi there!", "Hey, how can I help you?"]
  },
  {
    "tag": "goodbye",
    "patterns": ["Bye", "See you", "Goodbye"],
    "responses": ["Goodbye!", "See you later!", "Bye! Have a great day!"]
  }
]
```

## How to Run
Clone this repo or download the files.
Place your intents.json in the same folder as bert_chatbot.py.
Run the chatbot script:

```bash
python bert_chatbot.py
```
Chat with the bot in the terminal!
Type your message, and it will reply based on trained intents.

Press Enter (empty input) or type "exit" to quit.

## Future Improvements
- Add Streamlit UI for a web-based chatbot.
- Store chat history in CSV or database.
- Use advanced models like RoBERTa, DistilBERT, or even LLMs.
- Add Named Entity Recognition (NER) or slot-filling for smarter replies.




