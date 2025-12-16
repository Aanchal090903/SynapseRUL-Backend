package com.fleet.ai.model;

public class TelemetryRequest {

    private float engineRpm;
    private float oilPressure;
    private float fuelPressure;
    private float coolantPressure;
    private float oilTemp;
    private float coolantTemp;

    public float getEngineRpm() {
        return engineRpm;
    }

    public float getOilPressure() {
        return oilPressure;
    }

    public float getFuelPressure() {
        return fuelPressure;
    }

    public float getCoolantPressure() {
        return coolantPressure;
    }

    public float getOilTemp() {
        return oilTemp;
    }

    public float getCoolantTemp() {
        return coolantTemp;
    }
}
