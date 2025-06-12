import joblib
import os
import lzma

def compress_model_lzma(model_path):
    if not os.path.exists(model_path):
        print(f"❌ File not found: {model_path}")
        return

    print(f"🔄 Loading model from: {model_path}")
    model = joblib.load(model_path)

    # Clean attributes (optional)
    if hasattr(model, "X_train_"):
        print("🧹 Removing X_train_ for compression")
        del model.X_train_

    compressed_path = model_path.replace(".pkl", "_compressed_lzma.pkl.xz")

    with lzma.open(compressed_path, "wb") as f:
        joblib.dump(model, f, protocol=4)

    original_size = os.path.getsize(model_path) / (1024 * 1024)
    compressed_size = os.path.getsize(compressed_path) / (1024 * 1024)
    print(f"✅ Compressed (LZMA) model saved to: {compressed_path}")
    print(f"📉 Size reduced: {original_size:.2f} MB → {compressed_size:.2f} MB\n")

paths = [
    r"C:\Users\yaswa\OneDrive\Desktop\JOBLENS\model\recommendation\logistic_recommendation_model.pkl",
    r"C:\Users\yaswa\OneDrive\Desktop\JOBLENS\model\recommendation\knn_recommendation.pkl"
]

for p in paths:
    compress_model_lzma(p)

print("✅ LZMA compression completed!")
