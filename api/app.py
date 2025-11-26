from flask import Flask, request, jsonify, render_template
import torch
import pandas as pd
import numpy as np
import os
from models.neural_collaborative_filtering import NCF
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Global variables to hold model and data
model = None
user_encoder = None
item_encoder = None
movies_df = None

def load_model_and_data():
    global model, user_encoder, item_encoder, movies_df
    
    data_dir = "data/raw/ml-latest-small"
    if not os.path.exists(data_dir):
        print("Data directory not found.")
        return
        
    # Load data for encoders
    ratings = pd.read_csv(os.path.join(data_dir, "ratings.csv"))
    movies_df = pd.read_csv(os.path.join(data_dir, "movies.csv"))
    
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    ratings['user_id_enc'] = user_encoder.fit_transform(ratings['userId'])
    ratings['item_id_enc'] = item_encoder.fit_transform(ratings['movieId'])
    
    num_users = len(user_encoder.classes_)
    num_items = len(item_encoder.classes_)
    
    # Load model
    model = NCF(num_users, num_items)
    model_path = "models/ncf_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print("Model loaded successfully")
    else:
        print("Warning: Model file not found. Predictions will be random.")

# Load resources on startup
with app.app_context():
    load_model_and_data()

@app.route('/')
def index():
    return "Movie Recommendation API is running. Use /recommend endpoint."

@app.route('/recommend', methods=['POST'])
def recommend():
    global model, user_encoder, item_encoder, movies_df
    
    data = request.get_json()
    user_id = data.get('user_id')
    n_recommendations = data.get('n_recommendations', 10)
    
    if user_id not in user_encoder.classes_:
        return jsonify({"error": "User not found"}), 404
        
    user_idx = user_encoder.transform([user_id])[0]
    
    # Generate predictions for all items
    all_items = torch.arange(len(item_encoder.classes_))
    user_tensor = torch.full((len(all_items),), user_idx)
    
    with torch.no_grad():
        predictions = model(user_tensor, all_items).squeeze()
        
    # Get top K
    top_k_indices = torch.topk(predictions, n_recommendations).indices.numpy()
    top_k_item_ids = item_encoder.inverse_transform(top_k_indices)
    
    recommendations = []
    for item_id in top_k_item_ids:
        movie_info = movies_df[movies_df['movieId'] == item_id].iloc[0]
        recommendations.append({
            "movieId": int(item_id),
            "title": movie_info['title'],
            "genres": movie_info['genres']
        })
        
    return jsonify({"user_id": user_id, "recommendations": recommendations})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
