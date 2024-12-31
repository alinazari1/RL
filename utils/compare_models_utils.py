import numpy as np
from scipy.spatial.distance import cosine

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
