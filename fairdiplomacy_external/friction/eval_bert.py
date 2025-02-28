import json
import os
import random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

def extract_features(messages):
    data =[]
    count_total_msg = 0
    for msg in messages:
        if msg['phase'][-1] != 'M':
            continue
        if msg['sender'] == msg['recipient']:
            continue
        if msg['sender'] == 'ALL' or msg['recipient'] == 'ALL':
            continue
        count_total_msg+=1
        
        text = msg['message']
        label = 1 if 'sender_labels' in msg and not msg['sender_labels'] else 0

        # Extract friction features if available, selecting the entry with the highest sum
        if 'friction_info' in msg and msg['friction_info']:
            best_friction = max(
                msg['friction_info'],
                key=lambda x: sum([x.get('1_rule', 0), x.get('2_rule', 0), x.get('3_rule', 0)])
            )
            features = [best_friction.get('1_rule', 0), best_friction.get('2_rule', 0), best_friction.get('3_rule', 0)]
            # data.append((text, features, label))
        else:
            features = [-1, -1,-1 ]
        data.append((text, features, label))
    print(f'total_msg {count_total_msg}')
    return data

# Create PyTorch Dataset and DataLoader
class CustomDataset(Dataset):
    def __init__(self, text, num_features, attention_mask, labels):
        self.text = text
        self.num_features = num_features
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'text': self.text[idx],
            'num_features': self.num_features[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }


# Define a Fine-Tuned BERT Model with Numerical Features
class BERTWithNumericalFeatures(nn.Module):
    def __init__(self, num_numeric_features):
        super(BERTWithNumericalFeatures, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.nn_layers = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size + num_numeric_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, numeric_features):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_embedding = bert_output.pooler_output  # [CLS] token representation

        # Concatenate BERT embedding with numerical features
        combined_features = torch.cat((bert_embedding, numeric_features), dim=1)

        # Pass through NN layers
        return self.nn_layers(combined_features)

def test_with_files():
    
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', type=int, default=1000, help='Number of samples to use from the dataset')
    parser.add_argument('--epoch', type=int, default=10, help='epoch to eval')
    args = parser.parse_args()

    # List of uploaded files
    work_path = '/diplomacy_cicero/fairdiplomacy_external/'
    test_path = f"/diplomacy_cicero/fairdiplomacy_external/peskov_test_1K_messages.json"

    gold_amr_path = f"{work_path}/peskov_train_messages.json"    
    with open(gold_amr_path, "r") as f:
        gold_amr_data = json.load(f)
        
    # test_path = f"{work_path}/denis_1K_fn_only.json"
    with open(test_path, "r") as file:
        test_data = json.load(file)
        
    test_data = extract_features(test_data)
    
    binary_classification_data1 = extract_features(gold_amr_data)
    train_data = binary_classification_data1 

    # Separate text, numerical features, and labels for training
    texts_train, numeric_features_train, labels_train = zip(*train_data)
    numeric_features_train = np.array(numeric_features_train)
    labels_train = np.array(labels_train)

    # Normalize numerical features
    scaler = StandardScaler()
    numeric_features_train = scaler.fit_transform(numeric_features_train)

    # Directory to save models
    save_dir = f"{work_path}/saved_models/{args.sample_size}"
    # 1500 is the best one of ours
    os.makedirs(save_dir, exist_ok=True)

    best_loss = float('inf')

    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Assume test_data is available and prepared similarly
    texts_test, numeric_features_test, labels_test = zip(*test_data)
    numeric_features_test = np.array(numeric_features_test)
    labels_test = np.array(labels_test)

    # Normalize numerical features using the same scaler
    numeric_features_test = scaler.transform(numeric_features_test)

    # Convert test texts to tokenized tensors
    tokenized_texts_test = tokenizer(
        list(texts_test), padding=True, truncation=True, max_length=512, return_tensors="pt"
    )

    # Convert numerical features and labels to tensors for test set
    numeric_features_tensor_test = torch.tensor(numeric_features_test, dtype=torch.float32)
    labels_tensor_test = torch.tensor(labels_test, dtype=torch.float32).unsqueeze(1)
    
    test_dataset = CustomDataset(
        tokenized_texts_test['input_ids'], numeric_features_tensor_test, tokenized_texts_test['attention_mask'], labels_tensor_test
    )

    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    # Initialize the model
    model = BERTWithNumericalFeatures(num_numeric_features=numeric_features_test.shape[1])
    model.load_state_dict(torch.load(f"{save_dir}/best_model_epoch_{args.epoch}.pth"))  # Load the best model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    # Track FP and FN cases
    true_positives = []
    false_negatives = []
    false_positives = []
    true_negatives = []

    with torch.no_grad():
        correct = 0
        total = 0
        true_positive = 0
        false_positive = 0
        false_negative = 0
        true_negative = 0

        for idx, batch in enumerate(test_loader):
            input_ids = batch['text'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            num_features = batch['num_features'].to(device)
            labels = batch['labels'].to(device)

            predictions = model(input_ids, attention_mask, num_features)
            predicted_labels = (predictions >= 0.5).float()

            # Compute classification performance metrics
            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)

            # Identify FP and FN cases
            for i in range(labels.size(0)):
                if predicted_labels[i] == 1 and labels[i] == 1:  # True Positive
                    true_positives.append({
                        "text": texts_test[idx * test_loader.batch_size + i], 
                        "features": numeric_features_test[idx * test_loader.batch_size + i].tolist(), 
                        "true_label": labels[i].item(), 
                        "predicted_label": predicted_labels[i].item()
                    })
                elif predicted_labels[i] == 0 and labels[i] == 1:  # False Negative
                    false_negatives.append({
                        "text": texts_test[idx * test_loader.batch_size + i], 
                        "features": numeric_features_test[idx * test_loader.batch_size + i].tolist(), 
                        "true_label": labels[i].item(), 
                        "predicted_label": predicted_labels[i].item()
                    })
                elif predicted_labels[i] == 1 and labels[i] == 0:  # False positive
                    false_positives.append({
                        "text": texts_test[idx * test_loader.batch_size + i], 
                        "features": numeric_features_test[idx * test_loader.batch_size + i].tolist(), 
                        "true_label": labels[i].item(), 
                        "predicted_label": predicted_labels[i].item()
                    })
                elif predicted_labels[i] == 0 and labels[i] == 0:  # True negative
                    true_negatives.append({
                        "text": texts_test[idx * test_loader.batch_size + i], 
                        "features": numeric_features_test[idx * test_loader.batch_size + i].tolist(), 
                        "true_label": labels[i].item(), 
                        "predicted_label": predicted_labels[i].item()
                    })

                    
            true_positive += ((predicted_labels == 1) & (labels == 1)).sum().item()
            false_positive += ((predicted_labels == 1) & (labels == 0)).sum().item()
            false_negative += ((predicted_labels == 0) & (labels == 1)).sum().item()
            true_negative += ((predicted_labels == 0) & (labels == 0)).sum().item()

        # Print evaluation results
        print(f"Test Accuracy: {correct / total:.2%}")
        print(f"True Positives: {true_positive}")
        print(f"False Positives: {false_positive}")
        print(f"False Negatives: {false_negative}")
        print(f"True Negatives: {true_negative}")
        

    # stat = {'true_positives': true_positives, 'false_positives': false_positives, 'false_negatives': false_negatives,'true_negatives': true_negatives}
    stat = {'true_positives': true_positives, 'false_positives': false_positives, 'false_negatives': false_negatives}
    # output_file = f"{work_path}/true_positives/" + f"meta_first_2K_msg.json"
    output_file = f"{work_path}/results/" + f"peskov_test_1K_messages_prediction.json"
    with open(output_file, "w") as f:
        json.dump(stat, f, indent=4)