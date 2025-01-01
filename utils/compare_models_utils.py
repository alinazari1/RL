import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(CTP1, CTP2):
    """
    Compute the cosine similarity matrix between topics in CTP1 and CTP2.
    """
    n_topics = CTP1.shape[0]
    cosine_sim_matrix = np.zeros((n_topics, n_topics))
    
    for i in range(n_topics):
        for j in range(n_topics):
            cosine_sim_matrix[i, j] = 1 - cosine(CTP1[i], CTP2[j])  # Cosine similarity
    return cosine_sim_matrix

def calculate_absolute_difference(CTP1, CTP2):
    """
    Compute the Absolute Difference in Normalized Sums matrix between CTP1 and CTP2.
    """
    def normalize(vector):
        return vector / np.sum(vector)

    n_topics = CTP1.shape[0]
    abs_diff_matrix = np.zeros((n_topics, n_topics))
    
    for i in range(n_topics):
        for j in range(n_topics):
            norm1 = normalize(CTP1[i])
            norm2 = normalize(CTP2[j])
            abs_diff_matrix[i, j] = np.sum(np.abs(norm1 - norm2))
    return abs_diff_matrix

def calculate_entropy(topic_distributions):
    """
    Calculate entropy for each topic distribution.
    """
    def entropy(distribution):
        nonzero_probs = distribution[distribution > 0]  # Avoid log(0)
        return -np.sum(nonzero_probs * np.log(nonzero_probs))
    
    return np.array([entropy(topic) for topic in topic_distributions])

def compare_models(CTP1, CTP2):
    """
    Compare CTP1 and CTP2 using cosine similarity, absolute difference, and entropy changes.
    Return the current state and top 5 topics in CTP2 with the highest entropy changes.
    """
    # Cosine Similarity Matrix
    cosine_sim_matrix = calculate_cosine_similarity(CTP1, CTP2)
    
    # Absolute Difference in Normalized Sums Matrix
    abs_diff_matrix = calculate_absolute_difference(CTP1, CTP2)
    
    # Entropy for CTP1 and CTP2
    entropy_CTP1 = calculate_entropy(CTP1)
    entropy_CTP2 = calculate_entropy(CTP2)
    entropy_changes = entropy_CTP2 - entropy_CTP1
             
    return cosine_sim_matrix, abs_diff_matrix, entropy_changes

# Function to calculate the base reward
def calculate_base_reward(topic, documents, threshold=0.7):
    base_reward = 0
    d = len(documents)
    for doc in documents:
        similarity = calculate_cosine_similarity(topic, doc)
        if similarity > threshold:
            base_reward += similarity
    return base_reward / d

# Function to update Q-values based on entropy and base reward
def cal_q_values(Q, topics, documents, alpha=0.1, gamma=0.9, lambda_entropy=0.5):
    updated_q_values = []
    for topic in topics:
        # Calculate the entropy change for the topic
        entropy = calculate_entropy(topic['distribution'])  # Assuming topic['distribution'] holds the topic's word distribution
        
        # Calculate the base reward from cosine similarity
        base_reward = calculate_base_reward(topic['vector'], documents)
        
        # Calculate total reward
        reward = base_reward + lambda_entropy * entropy
        
        # Get the Q-value for this topic (previous Q-value + reward + max future Q-value)
        best_future_q = max([Q.get(t, 0) for t in topics])  # Max Q-value for future topics
        updated_q = (1 - alpha) * Q.get(topic['id'], 0) + alpha * (reward + gamma * best_future_q)
        
        updated_q_values.append((topic['id'], updated_q))
    
    # Sort topics by Q-value
    updated_q_values.sort(key=lambda x: x[1], reverse=True)
    
    # Return top 5 topics with highest Q-values
    return updated_q_values[:5]
