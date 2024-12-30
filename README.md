# Q-learning Topic Selection for Topic Modeling

This repository demonstrates the application of Q-learning to refine and select topics in a topic modeling task. The process is iterative, and each iteration involves applying domain-specific aspect-based refinement to improve topic models.

## Phases:

1. **Data Collection**: Collect documents and preprocess them based on expert-defined keywords.
2. **Applying the Method**: Use LDA to generate an initial topic model, then refine it iteratively with aspect-based keywords.
3. **Reinforcement Learning Process**: Use Q-learning to compare, select, and refine topics based on reward signals and entropy comparisons.
4. **Analysis**: Analyze the results using techniques like heatmap comparisons, cosine similarity, and technology vision identification.

## Requirements

- Python 3.x
- NumPy
- Scikit-learn

To install dependencies:
```bash
pip install numpy scikit-learn
