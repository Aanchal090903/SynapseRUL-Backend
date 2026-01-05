package com.fleet.ai.service;

import ai.onnxruntime.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.stereotype.Service;

import java.io.InputStream;
import java.util.List;
import java.util.Map;

@Service
public class InferenceService {

    private final OrtSession session;
    private final OrtEnvironment env;
    private final StandardScaler scaler;

    public InferenceService(OrtSession session, OrtEnvironment env) throws Exception {
        this.session = session;
        this.env = env;

        // ---------- Load scaler.json ----------
        ObjectMapper mapper = new ObjectMapper();
        InputStream is = getClass()
                .getClassLoader()
                .getResourceAsStream("models/scaler.json");

        if (is == null) {
            throw new RuntimeException("❌ scaler.json not found in resources/models/");
        }

        Map<String, Object> data = mapper.readValue(is, Map.class);

        List<Double> meanList = (List<Double>) data.get("mean");
        List<Double> scaleList = (List<Double>) data.get("scale");

        float[] mean = new float[meanList.size()];
        float[] scale = new float[scaleList.size()];

        for (int i = 0; i < mean.length; i++) {
            mean[i] = meanList.get(i).floatValue();
            scale[i] = scaleList.get(i).floatValue();
        }

        this.scaler = new StandardScaler(mean, scale);

        System.out.println("✅ StandardScaler loaded");
        System.out.println("✅ ONNX inputs: " + session.getInputNames());
        System.out.println("✅ ONNX outputs: " + session.getOutputNames());
    }

    public float predict(
            float engineRpm,
            float oilPressure,
            float fuelPressure,
            float coolantPressure,
            float oilTemp,
            float coolantTemp
    ) throws Exception {

        float[] raw = new float[]{
                engineRpm,
                oilPressure,
                fuelPressure,
                coolantPressure,
                oilTemp,
                coolantTemp
        };

        float[] scaled = scaler.transform(raw);
        float[][] input = new float[][]{scaled};

        // 🔥 ONNX input name (confirmed from Python)
        String inputName = "input";

        try (OnnxTensor tensor = OnnxTensor.createTensor(env, input)) {

            OrtSession.Result result =
                    session.run(Map.of(inputName, tensor));

            // 🔥 Output[1] = probabilities
            Object probsOutput = result.get(1).getValue();

            float p;

            if (probsOutput instanceof float[][] probs) {
                // probs[0][0] = healthy
                // probs[0][1] = fault
                p = probs[0][1];
            } else {
                throw new IllegalStateException(
                        "Unexpected ONNX probability output type: " + probsOutput.getClass()
                );
            }

            // Clamp for realism
            p = Math.max(0.01f, Math.min(0.99f, p));

            return p;
        }
    }


    public String resolveStatus(float p) {
        return p > 0.75 ? "FAULT" :
                p > 0.40 ? "NEEDS_SERVICE_SOON" :
                        "HEALTHY";
    }
}