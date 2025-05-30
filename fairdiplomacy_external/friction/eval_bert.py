import json
import os
import random
import argparse
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--sample_size', type=int, default=1000, help='Number of samples to use from the dataset')
parser.add_argument('--epoch', type=int, default=10, help='epoch to eval')
args = parser.parse_args()

# List of uploaded files
work_path = '/classifier_friction_dip'
test_path = f'{workpath}/samples_test_1K_scores_moves.json'
# test_path = f"/fs/nexus-scratch/wwongkam/classifier_friction_dip/sample_1K_messages_propose_2_detection_3_1_complete_with_scores.json"
    
with open(test_path, "r") as file:
    test_data = json.load(file)



def extract_features(messages, message_focus=None):
    count_lies = 0
    count_non_rl = 0
    max_non_rl = 100
    data =[]
    for msg in messages:
        text = f"{msg['sender']} sends to {msg['recipient']} with a message: {msg['message']}"
        label = 1 if not msg['sender_labels'] else 0
        scores = msg.get('scores',0)
        if label ==1:
            count_lies+=1

        # Extract friction features if available, selecting the entry with the highest sum
        if msg['friction_info']:
            best_friction = max(
                msg['friction_info'],
                key=lambda x: sum([x.get('1_rule', -1), x.get('2_rule', -1), x.get('3_rule', -1)])
            )
            features = [scores, best_friction.get('1_rule', -1), best_friction.get('2_rule', -1), best_friction.get('3_rule', -1)]
            # data.append((text, features, label))
        else:
            features = [scores, -1, -1,-1 ]
        
        # if len(msg['extracted_moves'])>0:
        data.append((text, features, label))
        # if message_focus is None:
        #     data.append((text, features, label))
        # else:
        #     if message_focus in msg['message']:
        #         data.append((text, features, label))
            
    return data

test_data = extract_features(test_data)

# Normalize numerical features
# scaler = StandardScaler()
scaler = joblib.load(f"{work_path}/saved_models/scaler.pkl")

save_dir = f"{work_path}/saved_models/score_bert_500"
# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(save_dir)

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

test_dataset = CustomDataset(
    tokenized_texts_test['input_ids'], numeric_features_tensor_test, tokenized_texts_test['attention_mask'], labels_tensor_test
)

test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

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

# Initialize the model
model = BERTWithNumericalFeatures(num_numeric_features=numeric_features_test.shape[1])
model.load_state_dict(torch.load(f"{save_dir}/best_model_epoch_10.pth"))  # Load the best model

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
        
        # print(f"input_ids: {input_ids}")
        # print(f"attention_mask: {attention_mask}")
        # print(f"num_features: {num_features}")
        # print(f"labels: {labels}")

        predictions = model(input_ids, attention_mask, num_features)
        predicted_labels = (predictions >= 0.5).float()
        # print(f"predictions: {predictions}")

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
# stat = {'true_positives': true_positives, 'false_positives': false_positives, 'false_negatives': false_negatives}
# # output_file = f"{work_path}/true_positives/" + f"meta_first_2K_msg.json"
# output_file = f"{work_path}/stat/" + f"meta_2K_2_predicted_orders.json"
# with open(output_file, "w") as f:
#     json.dump(stat, f, indent=4)
