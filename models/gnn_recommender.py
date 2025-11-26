import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import HeteroData
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class GNNRecommender(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, num_users, num_movies):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, hidden_channels)
        self.movie_emb = nn.Embedding(num_movies, hidden_channels)
        
        self.gnn = GNNRecommender(hidden_channels, out_channels)
        self.gnn = to_hetero(self.gnn, metadata, aggr='sum')
        
        self.classifier = nn.Linear(out_channels, 1)

    def forward(self, data):
        x_dict = {
          "user": self.user_emb(data["user"].node_id),
          "movie": self.movie_emb(data["movie"].node_id),
        } 
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        
        # dot product for link prediction 
        return x_dict

def create_hetero_data(data_dir):
    ratings = pd.read_csv(f"{data_dir}/ratings.csv")
    movies = pd.read_csv(f"{data_dir}/movies.csv")

    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    
    ratings['user_id_enc'] = user_encoder.fit_transform(ratings['userId'])
    ratings['movie_id_enc'] = movie_encoder.fit_transform(ratings['movieId'])
    
    data = HeteroData()
    
    # nodes
    data['user'].node_id = torch.arange(len(user_encoder.classes_))
    data['movie'].node_id = torch.arange(len(movie_encoder.classes_))
    
    # edges
    src = torch.tensor(ratings['user_id_enc'].values, dtype=torch.long)
    dst = torch.tensor(ratings['movie_id_enc'].values, dtype=torch.long)
    
    data['user', 'rates', 'movie'].edge_index = torch.stack([src, dst])
    
    # reverse edges
    data['movie', 'rated_by', 'user'].edge_index = torch.stack([dst, src])
    
    return data, len(user_encoder.classes_), len(movie_encoder.classes_)

if __name__ == "__main__":
    import os
    if os.path.exists("data/raw/ml-latest-small"):
        print("Creating Graph Data...")
        data, num_users, num_movies = create_hetero_data("data/raw/ml-latest-small")
        print(data)
        
        model = HeteroGNN(data.metadata(), hidden_channels=64, out_channels=32, num_users=num_users, num_movies=num_movies)
        print("Model initialized.")
        
        with torch.no_grad():
            out = model(data)
            print("Forward pass successful.")
            print(f"User embeddings shape: {out['user'].shape}")
    else:
        print("Data not found.")
