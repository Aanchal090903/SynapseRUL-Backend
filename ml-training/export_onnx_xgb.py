import pickle
import onnxmltools
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

# ----------------------------------------------------------
# Load trained XGBoost model
# ----------------------------------------------------------
with open("artifacts/xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

N_FEATURES = 6

# ----------------------------------------------------------
# Convert XGBoost → ONNX
# ----------------------------------------------------------
initial_type = [('input', FloatTensorType([None, N_FEATURES]))]

onnx_model = convert_xgboost(
    xgb_model,
    initial_types=initial_type
)

# ----------------------------------------------------------
# Save ONNX
# ----------------------------------------------------------
with open("artifacts/model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("✅ XGBoost ONNX model exported successfully")
