import joblib
import os

def compress_model(model_path, compress_level=3):
    if not os.path.exists(model_path):
        print(f"❌ File not found: {model_path}")
        return

    # Load the existing model
    print(f"🔄 Loading model from: {model_path}")
    model = joblib.load(model_path)

    # Generate compressed file path
    compressed_path = model_path.replace(".pkl", "_compressed.pkl")

    # Save compressed model
    joblib.dump(model, compressed_path, compress=compress_level)

    # Print size comparison
    original_size = os.path.getsize(model_path) / (1024 * 1024)
    compressed_size = os.path.getsize(compressed_path) / (1024 * 1024)
    print(f"✅ Compressed model saved to: {compressed_path}")
    print(f"📦 Size reduced: {original_size:.2f} MB → {compressed_size:.2f} MB\n")

# 🔧 Path to your Naïve Bayes Recommendation model
recommendation_model_path = r"C:\Users\yaswa\OneDrive\Desktop\JOBLENS\model\recommendation\naive_bayes_recommendation_model.pkl"

# 🔁 Compress only the recommendation model
compress_model(recommendation_model_path)

print("✅ Recommendation model compression completed!")
