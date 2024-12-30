import sys
from topic_modeling import data_collection, topic_model, rl_process, analysis

def main():
    # Phase 1: Data Collection
    domain = "Quantum Cryptography"
    search_keywords = data_collection.define_search_keywords(domain)
    corpus = data_collection.build_corpus(search_keywords)
    
    # Phase 2: Apply Topic Model Analysis
    TM = topic_model.LDA(corpus)
    CTP1, CTP2 = topic_model.initialize_topic_model(TM)
    
    # Start the episode loop
    rl_process.episode_loop(CTP1, CTP2)

if __name__ == "__main__":
    main()
