import pickle
import json

scaler = pickle.load(open("artifacts/scaler.pkl", "rb"))

scaler_data = {
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist()
}

with open("artifacts/scaler.json", "w") as f:
    json.dump(scaler_data, f, indent=2)

print("✅ scaler.json created")
