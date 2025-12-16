# 🚚 Cloud Telematics Platform with ONNX Inference for Fleet Analytics & Predictive Maintenance

## 📌 Project Overview

This project implements a **cloud-based telematics platform** for **fleet analytics and predictive maintenance**, designed using a **hybrid Edge–Cloud architecture**.

Vehicles act as **edge nodes**, generating real-time telemetry data, while all compute-intensive analytics—including machine learning inference and explainability—are centralized in the **cloud**. This architecture reflects real-world automotive and logistics platforms, where fleet-level context, historical trends, and regulatory requirements make cloud-based intelligence more practical than edge-only AI.

The system demonstrates how **ONNX-based inference** can be deployed in a **Spring Boot backend** to serve scalable, production-grade predictive maintenance models trained in Python.

---

## 🧠 Problem Statement: Cloud-Based vs Edge AI Predictive Maintenance

Predictive maintenance systems must decide **where intelligence should live**:

- **Edge devices** are constrained by hardware, power, and update complexity.
- **Cloud platforms** can aggregate fleet-wide data, store long-term history, and support complex analytics and explainability.

This project adopts a **hybrid approach**:
- The **edge** focuses on data generation and real-time monitoring.
- The **cloud** performs predictive maintenance, fleet analytics, and explainable AI.

---

## 🏗️ System Architecture

### High-Level Architecture

EDGE (Vehicle / ECU Simulator)
├─ Sensors (RPM, temperature, pressure, etc.)
├─ Noise & fault injection
└─ MQTT publish
↓
══════════════════════════════
☁️ CLOUD TELEMATICS PLATFORM
══════════════════════════════
├─ Spring Boot Ingestion Service
├─ ONNX Inference (Predictive Maintenance)
├─ Fleet Analytics & Business Rules
├─ Explainability Service (LIME)
└─ Fleet Dashboard & APIs



---

## 🚗 Edge Layer (Vehicle Simulation)

The edge layer represents realistic vehicle behavior:

- Sensor data generation (engine RPM, oil pressure, fuel pressure, coolant metrics)
- Fault pattern simulation (fuel system issues, overheating, oil pressure drops)
- Noise injection to mimic real-world sensor drift
- MQTT-based telemetry publishing

**Key file:**
- `simulation_mqtt.py`

The edge performs **no heavy machine learning inference**, reflecting real vehicle constraints.

---

## ☁️ Cloud Layer (Core Platform)

### 1️⃣ Telematics Ingestion

- Spring Boot service ingests real-time telemetry streams
- Designed for high-throughput, fleet-scale data ingestion
- Data persisted for historical analysis (PostgreSQL / TimescaleDB compatible)

---

### 2️⃣ ONNX-Based Predictive Maintenance

- Models are trained in Python and exported to **ONNX**
- Inference runs in **Spring Boot using ONNX Runtime (Java)**
- Outputs include:
  - Fault probability
  - Risk classification
  - Maintenance recommendations

This decouples **model training** from **production inference**, following enterprise best practices.

---

### 3️⃣ Fleet Analytics & Decision Logic

- Machine learning predictions are combined with domain-specific rules
- Enables safer, interpretable decisions (e.g., fuel pressure or coolant temperature thresholds)
- Supports fleet-level health monitoring and trend analysis

---

## 🧠 Explainable AI (LIME)

Predictive maintenance requires **trust and transparency**.

- A separate Python explainability service generates **LIME explanations**
- Provides human-readable insights into why a vehicle is flagged as high risk
- Supports compliance, debugging, and operator trust

Explainability is intentionally handled in the **cloud**, where computational resources are available.

---

## 📊 Fleet Dashboard

A live dashboard visualizes:

- Vehicle health status
- Fault probabilities
- Maintenance recommendations
- Sensor trends
- On-demand LIME explanations

The dashboard consumes **cloud APIs only** and does not run machine learning models locally.

---

## 🔐 Enterprise-Grade Features

- **Security:** OAuth2 / JWT-based authentication (Spring Security)
- **Observability:** Actuator, Micrometer, Prometheus-ready metrics
- **Resilience:** Fault tolerance and rate limiting
- **Scalability:** Designed for thousands of vehicles reporting concurrently

---

## 🤖 Machine Learning Workflow

### Training (Python)

- Random Forest
- XGBoost
- VotingClassifier (used as a robustness benchmark)

The ensemble model validates feature relevance and robustness under noisy conditions.

---

### Production Inference

- A single optimized XGBoost model is exported to **ONNX**
- Deployed in Spring Boot for scalable, production-grade inference

This balances **accuracy, latency, and deployability**.

---

## 🔍 Why Cloud-Based Inference?

Predictive maintenance benefits from:

- Fleet-level context
- Long-term historical trends
- Explainability (XAI)
- Easier model updates and monitoring

These requirements make **cloud-based intelligence** more practical than edge-only AI, while edge devices remain responsible for real-time data collection and immediate alerts.

---

## 🧩 Key Takeaway

> **Edge devices generate signals.  
> The cloud converts those signals into intelligence.**

This project demonstrates how modern predictive maintenance platforms are architected in practice.

---

## 🚀 Technologies Used

- **Backend:** Spring Boot, ONNX Runtime (Java)
- **ML Training:** Python, scikit-learn, XGBoost
- **Explainability:** LIME
- **Messaging:** MQTT
- **Database:** PostgreSQL / TimescaleDB
- **Dashboard:** Streamlit

---

## 📌 Conclusion

This project is a **Cloud Telematics Platform with ONNX Inference for Fleet Analytics & Predictive Maintenance**, implementing a realistic hybrid Edge–Cloud architecture aligned with industry practices.

It demonstrates not only machine learning capability, but also **system design judgment**, **deployability**, and **enterprise readiness**.


## Fleet Predictive Maintenance Platform

This repository contains:
- ML model training (XGBoost, RandomForest, ONNX export)
- Spring Boot ONNX inference service
- REST API for engine fault prediction
