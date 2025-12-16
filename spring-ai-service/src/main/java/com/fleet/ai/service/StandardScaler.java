package com.fleet.ai.service;

public class StandardScaler {

    private final float[] mean;
    private final float[] scale;

    public StandardScaler(float[] mean, float[] scale) {
        this.mean = mean;
        this.scale = scale;
    }

    public float[] transform(float[] input) {
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = (input[i] - mean[i]) / scale[i];
        }
        return output;
    }
}
