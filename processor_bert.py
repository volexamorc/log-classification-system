import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize models
model_embedding = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight embedding model
model_classification = joblib.load("training/models/log-classification.joblib")  # Classification model


def classify_with_bert(log_message, confidence_threshold=0.5):
    """
    Classify log message using BERT embeddings and trained classifier.
    Returns 'Unclassified' if maximum probability is below threshold.

    Args:
        log_message (str): Input log message to classify
        confidence_threshold (float): Minimum probability threshold (default: 0.5)

    Returns:
        str: Predicted label or 'Unclassified'
    """
    # Generate embedding (already handles batch input with [log_message])
    embeddings = model_embedding.encode([log_message])

    # Get prediction probabilities
    probabilities = model_classification.predict_proba(embeddings)[0]

    # Check confidence threshold
    if np.max(probabilities) < confidence_threshold:
        return "Unclassified"

    # Get predicted label
    predicted_label = model_classification.predict(embeddings)[0]
    return predicted_label


if __name__ == "__main__":
    test_logs = [
        "alpha.osapi_compute.wsgi.server - 12.10.11.1 - API returned 404 not found error",
        "GET /v2/3454/servers/detail HTTP/1.1 RCODE   404 len: 1583 time: 0.1878400",
        "System crashed due to drivers errors when restarting the server",
        "Hey bro, chill ya!",
        "Multiple login failures occurred on user 6454 account",
        "Server A790 was restarted unexpectedly during the process of data transfer"
    ]

    print("Log Classification Results:")
    print("=" * 50)
    for log in test_logs:
        label = classify_with_bert(log)
        print(f"{log[:60]}... -> {label}")  # Truncate long logs for display