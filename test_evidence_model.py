import joblib
import numpy as np

def load_model(model_dir):
    """Load the trained model and vectorizer."""
    model = joblib.load(f"{model_dir}/model.joblib")
    vectorizer = joblib.load(f"{model_dir}/vectorizer.joblib")
    return model, vectorizer

def predict(text, model, vectorizer):
    """Make a prediction using the model."""
    # Transform the input text
    X = vectorizer.transform([text])
    # Get prediction and probability
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]  # Probability of positive class
    return prediction, probability

def main():
    # Load model
    model_dir = "trained_model"
    model, vectorizer = load_model(model_dir)
    
    # Test cases
    test_cases = [
        "The company reported a 20% increase in revenue for Q1 2024.",
        "The weather is nice today.",
        "According to the financial report, the company's net profit margin increased by 15% year-over-year.",
        "The stock market is volatile.",
        "The quarterly earnings report shows a 30% growth in operating income.",
        "The sky is blue.",
        "The annual report indicates a 25% increase in market share.",
        "It's raining outside.",
        "The company's revenue grew by 40% compared to the previous quarter.",
        "The birds are chirping."
    ]
    
    print("\nTesting model predictions:")
    print("-" * 80)
    for text in test_cases:
        pred, prob = predict(text, model, vectorizer)
        print(f"\nText: {text}")
        print(f"Prediction: {'Positive' if pred == 1 else 'Negative'}")
        print(f"Confidence: {prob:.2%}")
        print("-" * 80)

if __name__ == "__main__":
    main() 