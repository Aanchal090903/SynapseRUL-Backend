package com.fleet.ai.controller;

import com.fleet.ai.model.TelemetryRequest;
import com.fleet.ai.service.InferenceService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/inference")
public class TelemetryController {

    private final InferenceService inferenceService;

    public TelemetryController(InferenceService inferenceService) {
        this.inferenceService = inferenceService;
    }

    @PostMapping
    public ResponseEntity<?> infer(@RequestBody TelemetryRequest req) {

        try {
            float prob = inferenceService.predict(
                    req.getEngineRpm(),
                    req.getOilPressure(),
                    req.getFuelPressure(),
                    req.getCoolantPressure(),
                    req.getOilTemp(),
                    req.getCoolantTemp()
            );

            Map<String, Object> response = new HashMap<>();
            response.put("faultProbability", prob);
            response.put("status", prob >= 0.5f ? "FAULT" : "HEALTHY");

            return ResponseEntity.ok(response);

        } catch (Exception e) {
            e.printStackTrace(); // 🔥 IMPORTANT

            Map<String, Object> error = new HashMap<>();
            error.put("error", "Inference failed");
            error.put("message", e.getMessage());

            return ResponseEntity.internalServerError().body(error);
        }
    }
}
