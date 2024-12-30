def save_model(model, filename):
    # Save model to file (e.g., pickle or joblib)
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved as {filename}")
