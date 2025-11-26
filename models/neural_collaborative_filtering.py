import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class NCFDataset(Dataset):
    def __init__(self, users, items, ratings):
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, hidden_layers=[64, 32]):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        layers = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        
        self.fc_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        x = torch.cat([user_emb, item_emb], dim=1)
        x = self.fc_layers(x)
        output = self.output_layer(x)
        return self.sigmoid(output)

def train_ncf(data_dir, epochs=5, batch_size=256, embedding_dim=32):
    ratings = pd.read_csv(f"{data_dir}/ratings.csv")
    
    # encode users and items
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    ratings['user_id_enc'] = user_encoder.fit_transform(ratings['userId'])
    ratings['item_id_enc'] = item_encoder.fit_transform(ratings['movieId'])
    
    num_users = ratings['user_id_enc'].nunique()
    num_items = ratings['item_id_enc'].nunique()
    
    train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
    
    train_dataset = NCFDataset(train_data['user_id_enc'].values, train_data['item_id_enc'].values, train_data['rating'].values)
    test_dataset = NCFDataset(test_data['user_id_enc'].values, test_data['item_id_enc'].values, test_data['rating'].values)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model, Loss, Optimizer
    model = NCF(num_users, num_items, embedding_dim=embedding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for users, items, ratings_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(users, items).squeeze()
            # Normalize ratings to 0-1 for sigmoid output if needed, or change model to output raw score
            # Here we assume ratings are 0.5-5.0. Let's normalize to 0-1 range for training with Sigmoid
            ratings_norm = ratings_batch / 5.0
            
            loss = criterion(outputs, ratings_norm)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
        
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for users, items, ratings_batch in test_loader:
            outputs = model(users, items).squeeze()
            ratings_norm = ratings_batch / 5.0
            loss = criterion(outputs, ratings_norm)
            test_loss += loss.item()
            
    print(f"Test Loss: {test_loss/len(test_loader):.4f}")
    
    torch.save(model.state_dict(), "models/ncf_model.pth")
    print("Model saved to models/ncf_model.pth")

if __name__ == "__main__":
    import os
    if os.path.exists("data/raw/ml-latest-small"):
        train_ncf("data/raw/ml-latest-small")
    else:
        print("Data not found. Run download.py first.")
