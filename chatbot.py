import json
import random
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, TextClassificationPipeline


def load_intents(path='intents.json'):
    with open(path, 'r') as file:
        intents = json.load(file)
    texts, tags = [], []
    for intent in intents:
        for pattern in intent['patterns']:
            texts.append(pattern)
            tags.append(intent['tag'])
    return texts, tags, intents

class IntentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

def train_model():
    texts, tags, intents = load_intents()
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(tags)

    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, encoded_labels, test_size=0.2, random_state=42
    )

    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt")
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, return_tensors="pt")

    
    train_dataset = IntentDataset(train_encodings, train_labels)
    val_dataset = IntentDataset(val_encodings, val_labels)


    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(tags)))

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy='epoch',
        logging_dir='./logs',
        save_total_limit=1,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    print("Training BERT model for intent classification...")
    trainer.train()

    return model, tokenizer, label_encoder, intents

def get_response(pipeline, label_encoder, intents, user_input):
    result = pipeline(user_input)[0]
    label_index = int(result['label'].split("_")[-1])
    tag = label_encoder.inverse_transform([label_index])[0]

    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, I didn't get that."

def chat():
    model, tokenizer, label_encoder, intents = train_model()
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=False)

    print("\nChatbot is ready! Type 'exit' or press Enter to quit.")
    while True:
        user_input = input("You: ").strip()
        if user_input == "":
            print("Exiting. Have a great day!")
            break
        response = get_response(pipe, label_encoder, intents, user_input)
        print("Bot:", response)

if __name__ == "__main__":
    chat()
