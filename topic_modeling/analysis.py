def compare_topic_models(CTP1, CTP2):
    """
    Compare two topic models (CTP1 and CTP2) using cosine similarity and entropy.
    
    Expert Analysis:
    - Review the heatmap to identify which topics in CTP1 and CTP2 show high similarity or large differences.
    - Significant differences indicate the presence of new or refined topics, suggesting novelty.
    - Minor changes or overlap might indicate that more iterations are needed.
    
    Returns:
        similarity_matrix (numpy.ndarray): The similarity matrix between CTP1 and CTP2.
    """
    similarity_matrix = cosine_similarity(CTP1, CTP2)
       
    # Expert Decision:
    # - If most values in the heatmap are close to 1, it's a sign of high overlap, which may suggest low novelty.
    # - If there are notable sections with low similarity (close to 0), these indicate areas of novelty.
    
    return similarity_matrix

def technology_vision(QCrypt_data, CTP2_Allwords):
    """
    Identify novel insights or patterns based on cosine similarity and entropy changes
    between QCrypt conference data and CTP2 Allwords.
    
    Expert Analysis:
    - Review the similarity between QCrypt documents and CTP2 topics.
    - If novel patterns are found (i.e., high cosine similarity with novel topics), this could indicate new directions for research or business insights.
    - If similarities are low, the expert may suggest further refinements (e.g., adding new aspect keywords).
    
    Returns:
        top_n_indices (numpy.ndarray): Indices of top N most similar topics/documents.
    """
    similarity_scores = cosine_similarity(QCrypt_data, CTP2_Allwords)
    
    # Identify the top N most similar topics/documents
    top_n_indices = np.argsort(similarity_scores, axis=1)[:, -5:]  # Top 5 most similar topics
    
    # Expert Decision:
    # - If the expert sees consistent patterns with low entropy across top N, they may conclude that novelty has been reached.
    # - If the top N shows large entropy or unclear patterns, it indicates that refinement is necessary.
       
    return top_n_indices

def fine_tune_topics(CTP1, CTP2, DocsCTP2, Patterns_Novelty):
    """
    Refine topics based on the results from topic analysis and identified patterns in DocsCTP2.
    
    Expert Analysis:
    - The expert analyzes how well the topics match the identified patterns.
    - If the refinement shows better clarity and consistency, they may decide to finalize the topics.
    - If further refinement is needed to capture more subtle patterns, additional iterations may be required.
    
    Returns:
        refined_topics (numpy.ndarray): Refined topics after further analysis and expert decision.
    """
    refined_topics = []
    
    for pattern in Patterns_Novelty:
        # Further refine topics based on the identified patterns
        refined_topic = np.mean([CTP1[pattern], CTP2[pattern]], axis=0)
        refined_topics.append(refined_topic)
    
    refined_topics = np.array(refined_topics)
       
    # Expert Decision:
    # - If the refined topics show clear, meaningful patterns, the expert may decide the episode should end (novelty reached).
    # - If the topics still seem unclear or need more adjustments, the expert may continue refining by injecting new aspect keywords.

    return refined_topics

