import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("artifacts/model.onnx")

for inp in sess.get_inputs():
    print("Input name:", inp.name, inp.shape, inp.type)

print("Outputs:", sess.get_outputs())

x = np.random.rand(1, 6).astype("float32")

out = sess.run(None, {"input": x})

print("ONNX output:", out)
