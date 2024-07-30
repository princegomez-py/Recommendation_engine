

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Compute collaborative filtering recommendations with index checks
def get_cf_recommendations(user_id, user_factors, item_factors, user_id_to_index, top_n=10):
    if user_id not in user_id_to_index:
        return []
    user_index = user_id_to_index[user_id]
    user_vector = user_factors[user_index]
    scores = np.dot(item_factors, user_vector)
    top_items = np.argsort(scores)[::-1][:top_n]
    return top_items

# Compute content-based filtering recommendations with index checks
def get_cb_recommendations(item_id, item_similarity, item_id_to_index, top_n=10):
    if item_id not in item_id_to_index:
        return []
    item_index = item_id_to_index[item_id]
    item_vector = item_similarity[item_index]
    top_items = np.argsort(item_vector)[::-1][:top_n]
    return top_items

# Combine collaborative and content-based recommendations
def get_hybrid_recommendations(user_id, item_id, user_factors, item_factors, item_similarity, user_id_to_index, item_id_to_index, alpha=0.5, top_n=10):
    cf_recommendations = get_cf_recommendations(user_id, user_factors, item_factors, user_id_to_index, top_n)
    cb_recommendations = get_cb_recommendations(item_id, item_similarity, item_id_to_index, top_n)
    
    hybrid_scores = {}
    for item in cf_recommendations:
        hybrid_scores[item] = hybrid_scores.get(item, 0) + alpha
    for item in cb_recommendations:
        hybrid_scores[item] = hybrid_scores.get(item, 0) + (1 - alpha)
    
    top_items = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [item[0] for item in top_items]

def load_and_preprocess_data(events_file, item_properties_file):
    # Load the dataset
    events = pd.read_csv(events_file)
    item_properties = pd.read_csv(item_properties_file)

    # Merge events with item properties
    events = events.merge(item_properties, on='itemid', how='left')

    # Create user-item interaction matrix
    user_item_matrix = events.pivot_table(index='visitorid', columns='itemid', values='event', aggfunc='count', fill_value=0)

    # Normalize the user-item interaction matrix
    user_item_matrix = user_item_matrix.div(user_item_matrix.sum(axis=1), axis=0)

    # Preprocess item properties
    item_properties['value'] = item_properties['value'].astype(str)
    item_properties['property_value'] = item_properties['property'] + '_' + item_properties['value']
    item_features = item_properties.pivot_table(index='itemid', columns='property_value', aggfunc='size', fill_value=0)

    # Create mappings from user_id and item_id to their respective indices
    user_id_to_index = {user_id: index for index, user_id in enumerate(user_item_matrix.index)}
    item_id_to_index = {item_id: index for index, item_id in enumerate(user_item_matrix.columns)}

    return user_item_matrix, item_features, user_id_to_index, item_id_to_index

def train_models(user_item_matrix, item_features):
    # Convert user-item matrix to sparse matrix
    sparse_user_item = csr_matrix(user_item_matrix.values)

    # Perform matrix factorization using TruncatedSVD
    svd = TruncatedSVD(n_components=50, random_state=42)
    user_factors = svd.fit_transform(sparse_user_item)
    item_factors = svd.components_.T

    # Compute cosine similarity between items
    item_similarity = cosine_similarity(item_features)

    return user_factors, item_factors, item_similarity

def main(user_id, item_id):
    # Load and preprocess data
    user_item_matrix, item_features, user_id_to_index, item_id_to_index = load_and_preprocess_data('events.csv', 'item_properties.csv')
    
    # Train models
    user_factors, item_factors, item_similarity = train_models(user_item_matrix, item_features)

    # Generate recommendations
    if user_id in user_id_to_index and item_id in item_id_to_index:
        top_recommendations = get_hybrid_recommendations(user_id, item_id, user_factors, item_factors, item_similarity, user_id_to_index, item_id_to_index)
        return top_recommendations
    else:
        return "User ID or Item ID not found in the dataset."

if __name__ == "__main__":
    # Example usage
    user_id = 1215488
    item_id = 36906
    recommendations = main(user_id, item_id)
    print(f"Top Recommendations for User {user_id} and Item {item_id}: {recommendations}")