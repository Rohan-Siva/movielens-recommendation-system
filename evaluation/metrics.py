import numpy as np

def precision_at_k(y_true, y_pred, k):
    relevant_items = set(y_true)
    recommended_items = set(y_pred[:k])
    intersection = relevant_items.intersection(recommended_items)
    return len(intersection) / k

def recall_at_k(y_true, y_pred, k):
    relevant_items = set(y_true)
    recommended_items = set(y_pred[:k])
    intersection = relevant_items.intersection(recommended_items)
    return len(intersection) / len(relevant_items) if len(relevant_items) > 0 else 0

def ndcg_at_k(y_true, y_pred, k):
    relevant_items = set(y_true)
    dcg = 0
    for i, item in enumerate(y_pred[:k]):
        if item in relevant_items:
            dcg += 1 / np.log2(i + 2)
    
    idcg = 0
    for i in range(min(len(relevant_items), k)):
        idcg += 1 / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0
