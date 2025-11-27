import torch
import pandas as pd
import os
import time
from models.neural_collaborative_filtering import NCF
from sklearn.preprocessing import LabelEncoder

def test_prediction_speed():
    data_dir = "data/raw/ml-latest-small"
    
    print("Loading data...")
    ratings = pd.read_csv(os.path.join(data_dir, "ratings.csv"))
    
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    ratings['user_id_enc'] = user_encoder.fit_transform(ratings['userId'])
    ratings['item_id_enc'] = item_encoder.fit_transform(ratings['movieId'])
    
    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)
    
    print(f"Users: {num_users}, Items: {num_items}")
    
    model = NCF(num_users, num_items)
    model.load_state_dict(torch.load("models/ncf_model.pth")) # warning: run this line after model training(ncf model script)
    model.eval()
    
    user_idx = 0
    all_items = torch.arange(num_items)
    user_tensor = torch.full((num_items,), user_idx)
    
    print("Starting prediction...")
    start_time = time.time()
    with torch.no_grad():
        predictions = model(user_tensor, all_items).squeeze()
    end_time = time.time()
    
    print(f"Prediction time for {num_items} items: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    test_prediction_speed()
