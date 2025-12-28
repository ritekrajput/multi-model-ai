# 🧠 Multimodal AI System for Depression Detection

## 📌 Project Overview

This project is a **multimodal AI-based depression detection system** designed to assist in the **early identification and severity assessment of depression** using multiple behavioral and contextual signals.

Unlike traditional binary classifiers, this system predicts **depression severity on a scale of 1–10** and separates **machine learning predictions** from **clinical interpretation**, making it suitable for healthcare-oriented decision support systems.

The project is developed as a **final-year engineering project** with an industry-style, scalable architecture.

---

## 🏗️ System Architecture

The project follows a layered backend architecture as shown below:

┌─────────────────────────────────────────────────────────────┐
│                    Mobile / Web Application                 │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Patient  │  │ Relative │  │ Results  │  │ History  │    │
│  │ Tests    │  │ Survey   │  │ Dashboard│  │ Tracking │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        API Gateway Layer                    │
│        (Authentication, Rate Limiting, Routing)             │
└─────────────────────────────────────────────────────────────┘
              │                     │                     │
              ▼                     ▼                     ▼
┌──────────────────┐   ┌──────────────┐   ┌─────────────────┐
│ Data Processing  │   │  AI / ML     │   │  Clinical Logic  │
│     Service      │   │   Engine     │   │     Service     │
└──────────────────┘   └──────────────┘   └─────────────────┘
              │                     │                     │
              └───────────────┬───────────────┬─────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        Database Layer                       │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Patient  │  │  Test    │  │  Model   │  │  Audit   │    │
│  │   DB     │  │ Results  │  │ Storage  │  │  Logs   │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────────────────┘


🔹 **Current development focuses on the AI/ML Engine, Inference Layer, and Clinical Logic**, which together form the core intelligence of the system.

---

## 🎯 Objectives

- Detect early signs of depression using **multimodal data**
- Incorporate **close-relative observations** to reduce bias in self-reporting
- Predict **depression severity (1–10)** instead of binary outcomes
- Separate **ML predictions** from **clinical decision logic**
- Build a system that is scalable, interpretable, and app-ready

---

## 🧩 Modalities Used

| Modality | Description | Status |
|--------|------------|--------|
| Patient Text | Self-reported test responses | ✅ Implemented |
| Relative Text | Behavioral change survey | ✅ Implemented |
| Patient Audio | Speech-based indicators | ⏸️ Designed, temporarily skipped |
| Patient Video | Facial behavior (OpenFace) | ⏸️ Designed, temporarily skipped |

⚠️ Audio and video pipelines are **implemented in code but temporarily skipped at the data level** to enable faster development and testing. They can be activated later without refactoring.

---

## 🧠 AI / ML Engine (Implemented)

### 🔹 Feature Extraction Pipelines
- **Text**: Transformer-based embeddings (BERT)
- **Audio**: OpenSMILE-based feature extractor (on-the-fly, optional)
- **Video**: OpenFace CSV-based facial behavior features

### 🔹 Models
- **Proposed Model**: Hierarchical Multimodal Fusion Model  
  - Cross-modal attention
  - Relative context integration
  - Regression output (severity score 1–10)

- **Baseline Model**: MFFNC (Multi-Feature Fusion Neural Classifier)  
  - Flat feature concatenation
  - Used only for performance comparison

### 🔹 Training
- Shared training and validation loop
- Loss function: `SmoothL1Loss`
- Metrics: MAE, RMSE
- Currently trained using text-based inputs

---

## 🔍 Inference Layer (Implemented)

The inference layer:
- Loads the trained model
- Accepts multimodal inputs
- Produces a **raw severity score**
- Clamps output to **1–10** during inference

This layer forms the boundary between **ML computation** and **application logic**.

---

## 🏥 Clinical Logic Service (Implemented)

Clinical logic is intentionally separated from machine learning.

It:
- Converts severity score into categories:
  - **Early**
  - **Moderate**
  - **Severe**
- Generates **risk flags** (e.g., high risk, social withdrawal, negative cognition)
- Provides interpretable outputs suitable for healthcare workflows

---


## 🧪 Current Status

- ✅ End-to-end training works (text-based)
- ✅ Inference and clinical logic fully functional
- ✅ Audio and video safely skipped using zero vectors
- (Audio and video pat will be done during frontend part)
- ❌ API Gateway, Database, and Frontend not yet implemented

This is an **intentional development stage**, not a limitation.

---

## 🚧 Planned Next Steps

1. **FastAPI Backend**
   - Expose inference and clinical logic as REST APIs
   - Align with API Gateway layer

2. **Database Integration**
   - Patient records
   - Test history
   - Severity trend tracking

3. **Enable Audio & Video**
   - Activate existing pipelines
   - No major code changes required

4. **Frontend Application**
   - Patient test interface
   - Relative survey
   - Results dashboard

---

## 🧠 Key Design Principles

- Multimodal by design
- Clinical interpretability over black-box predictions
- Clear separation of concerns
- Scalable, extensible architecture
- Academic and industry-ready implementation

---

## 📜 Disclaimer

This project is developed for **academic and research purposes** and is not intended to replace professional medical diagnosis.

---

