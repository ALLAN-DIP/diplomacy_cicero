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

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--sample_size', type=int, default=1000, help='Number of samples to use from the dataset')
args = parser.parse_args()

# List of uploaded files
work_path = '/fs/nexus-scratch/wwongkam/classifier_friction_dip'
file_paths = [
    f"{work_path}/denis_state_detection_weights3_1/game1_amr_proposals.json",
    f"{work_path}/denis_state_detection_weights3_1/game2_amr_proposals.json",
    f"{work_path}/denis_state_detection_weights3_1/game3_amr_proposals.json",
    f"{work_path}/denis_state_detection_weights3_1/game4_amr_proposals.json",
    f"{work_path}/denis_state_detection_weights3_1/game5_amr_proposals.json",
    f"{work_path}/denis_state_detection_weights3_1/game6_amr_proposals.json",
    f"{work_path}/denis_state_detection_weights3_1/game7_amr_proposals.json",
    f"{work_path}/denis_state_detection_weights3_1/game8_amr_proposals.json",
    f"{work_path}/denis_state_detection_weights3_1/game9_amr_proposals.json",
    f"{work_path}/denis_state_detection_weights3_1/game10_amr_proposals.json",
    f"{work_path}/denis_state_detection_weights3_1/game11_amr_proposals.json",
    f"{work_path}/denis_state_detection_weights3_1/game12_amr_proposals.json",
]
test_path = f"{work_path}/sample_1K_messages_propose_2_detection_3_1_complete.json"
with open(test_path, "r") as file:
    test_data = json.load(file)

gold_amr_path = f"{work_path}/denis_train_messages_detection_1.json"    
with open(gold_amr_path, "r") as f:
    gold_amr_data = json.load(f)
    

# # Function to extract relevant data
# def extract_data_from_files(file_paths, test_data):
#     extracted_data = []
#     lie_features = []
#     truth_features = []
#     lie_no_features = []
#     truth_no_features = []
#     for file_path in file_paths:
#         with open(file_path, "r") as file:
#             data = json.load(file)

#         for phase in data.get("phases", []):
#             for msg in phase.get("messages", []):
#                 text = msg["message"]
#                 if any(text == m['message'] for m in test_data):
#                    continue
#                 label = 1 if not msg["sender_labels"] else 0

#                 # Extract friction features if available, selecting the entry with the highest sum
#                 if msg.get("friction_info"):
#                     best_friction = max(
#                         msg["friction_info"],
#                         key=lambda x: sum([x.get('1_rule', 0), x.get('2_rule', 0), x.get('3_rule', 0)]),
#                         default={"1_rule": 0.0, "2_rule": 0.0, "3_rule": 0.0}
#                     )
#                     features = [best_friction.get('1_rule', 0), best_friction.get('2_rule', 0), best_friction.get('3_rule', 0)]
#                     if label ==1:
#                         lie_features.append((text, features, label))
#                     else:
#                         truth_features.append((text, features, label))
#                 else:
#                     features = [-1, -1,-1 ]
#                     if label ==1:
#                         lie_no_features.append((text, features, label))
#                     else:
#                         truth_no_features.append((text, features, label))

#     return lie_features, truth_features, lie_no_features, truth_no_features

# Function to extract relevant data
def extract_data_from_files(file_paths, test_data):
    extracted_data = []
    for file_path in file_paths:
        with open(file_path, "r") as file:
            data = json.load(file)

        for phase in data.get("phases", []):
            for msg in phase.get("messages", []):
                text = msg["message"]
                if any(text == m['message'] for m in test_data):
                   continue
                label = 1 if not msg["sender_labels"] else 0

                # Extract friction features if available, selecting the entry with the highest sum
                if msg.get("friction_info"):
                    best_friction = max(
                        msg["friction_info"],
                        key=lambda x: sum([x.get('1_rule', 0), x.get('2_rule', 0), x.get('3_rule', 0)]),
                        default={"1_rule": 0.0, "2_rule": 0.0, "3_rule": 0.0}
                    )
                    features = [best_friction.get('1_rule', 0), best_friction.get('2_rule', 0), best_friction.get('3_rule', 0)]
                    
                else:
                    features = [-1, -1,-1 ]
                extracted_data.append((text, features, label))
                
    return extracted_data

def extract_features(messages):
    count_lies = 0
    count_non_rl = 0
    max_non_rl = 100
    data =[]
    for msg in messages:
        text = msg['message']
        label = 1 if not msg['sender_labels'] else 0
        if label ==1:
            count_lies+=1

        # Extract friction features if available, selecting the entry with the highest sum
        if msg['friction_info']:
            best_friction = max(
                msg['friction_info'],
                key=lambda x: sum([x.get('1_rule', 0), x.get('2_rule', 0), x.get('3_rule', 0)])
            )
            features = [best_friction.get('1_rule', 0), best_friction.get('2_rule', 0), best_friction.get('3_rule', 0)]
          
        else:
            features = [-1, -1,-1 ]
        data.append((text, features, label))
            
    return data

sample_size = 1000
# Extract data from all files
# lie_features, truth_features, lie_no_features, truth_no_features = extract_data_from_files(file_paths, test_data)
# lie_features = random.sample(lie_features, min(int(args.sample_size/4), len(lie_features)))
# truth_features = random.sample(truth_features, min(int(args.sample_size/4), len(truth_features)))
# lie_no_features = random.sample(lie_no_features, min(int(args.sample_size/4), len(lie_no_features)))
# truth_no_features = random.sample(truth_no_features, min(int(args.sample_size/4), len(truth_no_features)))


# binary_classification_data2 = lie_features + truth_features + lie_no_features + truth_no_features
binary_classification_data2 = extract_data_from_files(file_paths, test_data)
    
binary_classification_data1 = extract_features(gold_amr_data)

test_data = extract_features(test_data)


# Directory to save models
save_dir = f"{work_path}/saved_models/bert_{args.sample_size}"
os.makedirs(save_dir, exist_ok=True)

best_loss = float('inf')

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Use extracted data as train_data and assume test_data is predefined
train_data = binary_classification_data1 +binary_classification_data2 # Extracted from the files
test_data = test_data

# Separate text, numerical features, and labels for training
texts_train, numeric_features_train, labels_train = zip(*train_data)
numeric_features_train = np.array(numeric_features_train)
labels_train = np.array(labels_train)

# Normalize numerical features
scaler = StandardScaler()
numeric_features_train = scaler.fit_transform(numeric_features_train)

# Convert texts to tokenized tensors
tokenized_texts_train = tokenizer(
    list(texts_train), padding=True, truncation=True, max_length=512, return_tensors="pt"
)

# Convert numerical features and labels to tensors
numeric_features_tensor_train = torch.tensor(numeric_features_train, dtype=torch.float32)
labels_tensor_train = torch.tensor(labels_train, dtype=torch.float32).unsqueeze(1)

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

# Create train and test datasets
train_dataset = CustomDataset(
    tokenized_texts_train['input_ids'], numeric_features_tensor_train, tokenized_texts_train['attention_mask'], labels_tensor_train
)
test_dataset = CustomDataset(
    tokenized_texts_test['input_ids'], numeric_features_tensor_test, tokenized_texts_test['attention_mask'], labels_tensor_test
)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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
    
class NumericalFeatures(nn.Module):
    def __init__(self, num_numeric_features):
        super(NumericalFeatures, self).__init__()
        self.nn_layers = nn.Sequential(
            nn.Linear(num_numeric_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, numeric_features):
        # Pass through NN layers
        return self.nn_layers(numeric_features)

class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.nn_layers = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_embedding = bert_output.pooler_output  # [CLS] token representation

        # Pass through NN layers
        return self.nn_layers(bert_embedding)

# Initialize the model
model = BERT()

# model = NumericalFeatures(num_numeric_features=numeric_features_train.shape[1])

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)  # Lower learning rate for fine-tuning BERT

# Training loop
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        input_ids = batch['text'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        # num_features = batch['num_features'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        # outputs = model(input_ids, attention_mask, num_features)
        # outputs = model(num_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        # Save model if it achieves a new best loss
    if avg_loss < best_loss:
        best_loss = avg_loss
        model_path = os.path.join(save_dir, f"best_model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved at epoch {epoch+1} with loss: {best_loss:.4f}")

# Track FP and FN cases
false_positives = []
false_negatives = []

with torch.no_grad():
    correct = 0
    total = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    for idx, batch in enumerate(test_loader):
        input_ids = batch['text'].to(device)
        # attention_mask = batch['attention_mask'].to(device)
        # num_features = batch['num_features'].to(device)
        labels = batch['labels'].to(device)

        predictions = model(input_ids, attention_mask)
        # predictions = model(input_ids, attention_mask, num_features)
        # predictions = model(num_features)
        predicted_labels = (predictions >= 0.5).float()

        # Compute classification performance metrics
        correct += (predicted_labels == labels).sum().item()
        total += labels.size(0)

        # Identify FP and FN cases
        for i in range(labels.size(0)):
            if predicted_labels[i] == 1 and labels[i] == 0:  # False Positive
                false_positives.append({
                    # "text": texts_test[idx * test_loader.batch_size + i], 
                    "features": numeric_features_test[idx * test_loader.batch_size + i], 
                    "true_label": labels[i].item(), 
                    "predicted_label": predicted_labels[i].item()
                })
            elif predicted_labels[i] == 0 and labels[i] == 1:  # False Negative
                false_negatives.append({
                    # "text": texts_test[idx * test_loader.batch_size + i], 
                    "features": numeric_features_test[idx * test_loader.batch_size + i], 
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

    # Print a few false positive and false negative cases for inspection
    print("\nFalse Positives (Predicted 1, Actual 0):")
    for i, case in enumerate(false_positives[:5]):  # Show first 5 FP cases
        print(f"  {i+1}. Text: {case['text']}")
        # print(f"     Features: {case['features']}")
        print(f"     True Label: {case['true_label']}, Predicted Label: {case['predicted_label']}\n")

    print("\nFalse Negatives (Predicted 0, Actual 1):")
    for i, case in enumerate(false_negatives[:5]):  # Show first 5 FN cases
        print(f"  {i+1}. Text: {case['text']}")
        # print(f"     Features: {case['features']}")
        print(f"     True Label: {case['true_label']}, Predicted Label: {case['predicted_label']}\n")