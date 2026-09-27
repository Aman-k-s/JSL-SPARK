# JSL-SPARK: Steel Surface Quality & Root-Cause Platform

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?logo=python)](https://www.python.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8-Nano%20(NEU--DET)-00FFFF?logo=ultralytics)](https://github.com/ultralytics/ultralytics)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange?logo=databricks)](https://www.trychroma.com/)
[![LLM Engine](https://img.shields.io/badge/LLM-Groq%20Llama--3.3--70B-f55036?logo=groq)](https://groq.com/)
[![Framework](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Presentation](https://img.shields.io/badge/Slide%20Deck-Google%20Drive-4285F4?logo=googledrive&logoColor=white)](https://drive.google.com/file/d/1-RZ6eYCOYMQSU7XvfYYbCbSNKjrXydqx/view?usp=sharing)
[![Demo Video](https://img.shields.io/badge/Demo%20Video-YouTube-red?logo=youtube&logoColor=white)](https://youtu.be/r08QByG-3Pk)
[![License](https://img.shields.io/badge/License-Proprietary%20%2F%20JSL-lightgrey)]()

An automated end-to-end industrial inspection and root-cause analysis platform for steel strip surface defects. Built on fine-tuned **YOLOv8** object detection, **Temperature Scaling** statistical calibration, **morphological defect fingerprinting**, **ChromaDB vector retrieval** over peer-reviewed metallurgical failure literature, deterministic **operational decisioning**, **One-Class SVM anomaly detection**, and a conversational **metallurgical RAG assistant**.

> **Project Media & Resources:**
> - **Presentation Slide Deck (PPT):** [Google Drive Link](https://drive.google.com/file/d/1-RZ6eYCOYMQSU7XvfYYbCbSNKjrXydqx/view?usp=sharing)
> - **Full Video Walkthrough:** [YouTube Demonstration](https://youtu.be/r08QByG-3Pk)

---

## Table of Contents
1. [Executive Summary & Problem Statement](#executive-summary--problem-statement)
2. [Project Demonstration & Presentation Links](#project-demonstration--presentation-links)
3. [End-to-End System Architecture](#end-to-end-system-architecture)
4. [Key Platform Capabilities](#key-platform-capabilities)
   - [Deep Vision & Spatial Fingerprinting](#1-deep-vision--spatial-fingerprinting)
   - [Statistical Calibration (Temperature Scaling)](#2-statistical-confidence-calibration)
   - [Out-of-Distribution (OOD) Protection](#3-out-of-distribution-ood-detection)
   - [Metallurgical Knowledge Base & Vector RAG](#4-metallurgical-knowledge-base--vector-rag)
   - [Deterministic Operational Decision Engine](#5-deterministic-operational-decision-engine)
   - [Process Telemetry Correlation](#6-process-telemetry-correlation)
   - [Continuous Domain Feedback Loop](#7-continuous-domain-feedback-loop)
   - [Conversational Technical Assistant](#8-conversational-technical-assistant)
5. [Supported Defect Classes](#supported-defect-classes)
6. [Operational Decision Matrix](#operational-decision-matrix)
7. [Repository & Codebase Structure](#repository--codebase-structure)
8. [Installation & Local Quickstart](#installation--local-quickstart)
9. [Streamlit Cloud Deployment Guide](#streamlit-cloud-deployment-guide)
10. [Test Suite Execution](#test-suite-execution)
11. [Authors & Attribution](#authors--attribution)

---

## Project Demonstration & Presentation Links

| Resource | Link | Description |
|---|---|---|
| **Presentation Deck (PPT)** | [Google Drive Presentation](https://drive.google.com/file/d/1-RZ6eYCOYMQSU7XvfYYbCbSNKjrXydqx/view?usp=sharing) | High-level system architecture, problem statement, metallurgical RAG rationale, and operational results. |
| **Platform Video Demonstration** | [YouTube Video Walkthrough](https://youtu.be/r08QByG-3Pk) | Live demonstration of the automated inspection pipeline, detection visualization, grounded diagnosis, and interactive assistant. |

---

## Executive Summary & Problem Statement

In high-speed steel strip production, optical inspection systems traditionally output raw bounding boxes and class names (e.g., *"inclusion 82%"*). While useful for defect counting, this level of output leaves critical operational and metallurgical questions unanswered:
- **Overconfidence Risk:** Raw neural network confidences are miscalibrated, often asserting high certainty on ambiguous surface markings.
- **Unknown Root Cause:** Mill supervisors cannot determine whether a flaw originated in continuous casting (tramp elements, mold flux entrapment), reheat furnaces (oxide scale buildup), or mechanical mill contact (worn side guides, roll spalling).
- **Subjective Disposition:** Operators lack deterministic, repeatable rules for whether to continue rolling, route to surface grinding, or halt the mill to prevent roll damage.

**JSL-SPARK bridges the gap between computer vision and physical metallurgy.** It captures the spatial morphology of the defect, queries a curated knowledge base of metallurgical failure mechanisms via vector RAG, synthesizes a grounded investigation action, and executes a deterministic operational disposition in milliseconds.

---

## End-to-End System Architecture

```
+-----------------------------------------------------------------------------------+
|                           1. SPECIMEN ACQUISITION                                 |
|   - Real-Time Optical Strip Cameras / Front-Page Upload / 6-Class Validation      |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        2. VISION & PERCEPTION ENGINE                              |
|   - YOLOv8 Nano (`models/best.pt` fine-tuned on NEU-DET 200x200 surface crops)    |
|   - Bounding Box Localization, Defect Classification, Raw Detection Confidence    |
+-------------------+---------------------+--------------------+--------------------+
                    |                     |                    |
                    v                     v                    v
+-----------------------+ +--------------------+ +----------------------------------+
| Temperature Scaling   | | Spatial & Pattern  | | One-Class SVM Anomaly Node       |
| Calibrator (T=2.397)  | | Profiler           | | Evaluates pooled SPPF backbone   |
| Reduces ECE to 15.2%  | | Aspect ratio, edge | | embeddings; flags non-conforming |
|                       | | vs center, area %, | | defect distributions (> 0.65)    |
|                       | | severity score     | |                                  |
+-----------+-----------+ +---------+----------+ +-----------------+----------------+
            |                       |                              |
            +-----------+-----------+                              |
                        |                                          |
                        v                                          |
+-------------------------------------------------------+          |
|         3. METALLURGICAL VECTOR RAG CORE              |          |
|   - ChromaDB Vector Store (`all-MiniLM-L6-v2` dense)  |          |
|   - 48 Curated Passages across 6 Defect Classes       |          |
|   - 11 Peer-Reviewed Metallurgical Sources (S1-S11)   |          |
|   - Dense Cosine Similarity Retrieval & Filtering     |          |
+---------------------------+---------------------------+          |
                            |                                      |
                            v                                      |
+-------------------------------------------------------+          |
|        4. GROUNDED LLM DIAGNOSIS ENGINE               |          |
|   - Groq API (Llama-3.3-70B-Versatile)                |          |
|   - Metallurgical Diagnostic Policy:                  |          |
|     * Probable-Origin Hypothesis                      |          |
|     * Concrete Recommended Investigation Action       |          |
|     * Cited Passage ID & Technical Literature Source  |          |
+---------------------------+---------------------------+          |
                            |                                      |
                            v                                      |
+-------------------------------------------------------+          |
|     5. OPERATIONAL DECISION & TELEMETRY ENGINE        |          |
|   - Deterministic Triage: ACTION: ESCALATE / GRIND /  |          |
|     INSPECT with Auditable Rule Logging               |          |
|   - Process Telemetry Correlator (Reheat Temp, Speed, |          |
|     Descaling Pressure, Steel Grade Kinetics)         |          |
+---------------------------+---------------------------+          |
                            |                                      |
                            v                                      v
+-----------------------------------------------------------------------------------+
|                     6. ENTERPRISE OPERATOR INTERFACE                              |
|   - Recommended Operational Disposition (Top of Results)                          |
|   - Spatial Annotation Canvas & Defect Metric Summary Bar                         |
|   - Probable-Origin Hypothesis & Recommended Plant Investigation                  |
|   - Interactive Metallurgical RAG Assistant (Contextualized Q&A)                  |
|   - Operator Validation & Active ChromaDB Re-Indexing Loop                        |
+-----------------------------------------------------------------------------------+
```

---

## Key Platform Capabilities

### 1. Deep Vision & Spatial Fingerprinting
- **Detector:** Fine-tuned YOLOv8 Nano architecture trained on the NEU-DET steel surface database.
- **Morphological Profiler (`src/characterize.py`):**
  - **Aspect Ratio ($w/h$):** Distinguishes *elongated* defects (rolling direction streaks) from *compact* defects (localized pits or inclusions).
  - **Strip Location:** Identifies whether defects reside on strip edges (outer 15% strip boundary) vs. central strip body.
  - **Affected Area Coverage:** Precise surface percentage calculated against total image dimensions.
  - **Pattern Analyzer:** Classifies defect distribution as *isolated* or *repetitive* across the strip surface.
  - **Calibrated Severity Score:** Computes a composite metric (`Low`, `Medium`, `High`) based on calibrated probability and geometric footprint.

### 2. Statistical Confidence Calibration
Modern deep neural networks often exhibit significant overconfidence. JSL-SPARK incorporates a **post-hoc Temperature Scaling** module (`src/calibrate.py`):
- Optimization: Logit scaling parameterized by scalar temperature parameter $T = 2.3972$, fitted via Cross-Entropy loss minimization on the validation set.
- **Expected Calibration Error (ECE):** Reduced from **23.57%** (uncalibrated) down to **15.21%** (calibrated), a **35.5% relative error reduction**.
- Ensures operational thresholds reflect true statistical accuracy rather than inflated softmax probabilities.

### 3. Out-of-Distribution (OOD) Detection
- **Model:** One-Class Support Vector Machine (SVM) trained on pooled penultimate backbone convolutional feature maps (`src/anomaly.py`).
- **Function:** If an optical flaw produces low classification confidence alongside elevated feature space divergence (anomaly score $\ge 0.65$), the system triggers an **Out-of-Distribution Alert Banner**, preventing misclassification of unseen contaminants or sensor artifacts.

### 4. Metallurgical Knowledge Base & Vector RAG
- **Curated Literature Index:** 48 detailed metallurgical failure passages covering all 6 surface defect classes, grounded in 11 academic and technical publications (`S1` through `S11`).
- **Dense Embeddings:** Indexed in ChromaDB using `sentence-transformers/all-MiniLM-L6-v2`.
- **Diagnostic Policy:** Rather than asserting definitive root causes from images alone (which is physically invalid without destructive testing), the Groq Llama-3.3-70B model formulates **probable-origin hypotheses**, cites exact knowledge base passages, and prescribes **actionable plant investigation protocols** (e.g., checking descaling nozzle spray patterns or inspecting mill stand side guides).

### 5. Deterministic Operational Decision Engine
- Evaluates defect severity, surface area percentage, defect class, and vector retrieval similarity to recommend immediate mill actions (`ACTION: ESCALATE`, `ACTION: GRIND`, `ACTION: INSPECT`).
- Positioned prominently at the top of inspection results for rapid operator action.

### 6. Process Telemetry Correlation
- Correlates defect fingerprints with mill process variables (`src/process_correlate.py`):
  - Furnace Reheat Temperature & Atmosphere ($O_2$ excess)
  - Descaling Header Pressure (bar)
  - Finishing Mill Stand Rolling Speed & Roll Campaign Tonnage
  - Steel Grade & Chemistry (e.g., tramp copper content)
- Highlights anomalous process parameter deviations associated with observed flaw types.

### 7. Continuous Domain Feedback Loop
- **Operator Confirmation:** Validates automated diagnoses, logging an immutable audit record to `confirmations.jsonl`.
- **Domain Correction:** Plant metallurgists can submit corrective root-cause notes through a built-in popover form. Corrections are written to `corrections.jsonl` and immediately re-indexed into the ChromaDB vector database without taking down the server.

### 8. Conversational Technical Assistant
- Dedicated second tab in the application powered by RAG (`src/rag_assistant.py`).
- Pre-loaded with quick-inquiry buttons (e.g., descaling pressure requirements, side guide wear mechanisms, tramp element embrittlement).
- **Session Context Sharing:** Automatically attaches the active specimen's defect class, severity, and bounding box coordinates into chat prompts for seamless deep-dive Q&A.

---

## Supported Defect Classes

The platform supports all 6 canonical steel surface defect categories defined in the NEU-DET benchmark:

| Defect Class | Visual Description | Typical Metallurgical Origin |
|---|---|---|
| **Inclusion** | Dark elongated or granular non-metallic streaks aligned in rolling direction | Deoxidation products ($Al_2O_3$), ladle slag carryover, or mold flux entrapment during continuous casting |
| **Patches** | Irregular, localized, dark oxidized surface patches | Uneven primary/secondary scale removal, localized roll surface spalling, or hydraulic nozzle clogging |
| **Pitted Surface** | Clustered localized shallow depressions across strip surface | Over-pickling, acid entrapment, or severe roll surface micro-spalling transferred under pressure |
| **Rolled-in Scale**| Dark embedded iron oxide flakes pressed flush into the matrix | Inadequate primary descaling pressure (< 150 bar) or delayed secondary descaling before finishing stands |
| **Scratches** | Sharp, continuous longitudinal linear gouges | Mechanical contact with worn mill side guides, broken looper rollers, or misaligned table aprons |
| **Crazing** | Intricate network of fine, shallow surface micro-cracks | Thermal shock from inadequate roll cooling, roll thermal fatigue, or tramp copper hot shortness |

---

## Operational Decision Matrix

The deterministic operational logic in `src/decide.py` enforces the following auditable triage rules:

```
[Defect Fingerprint + Diagnostic Score]
                 │
                 ├── High Severity AND Area >= 15% ──────────────► ACTION: ESCALATE (Rule 1)
                 │   OR Critical Class (Crazing/Inclusion) >= 8%
                 │
                 ├── High Severity AND Diagnosis Conf < 0.55 ────► ACTION: ESCALATE (Rule 2)
                 │   (Ambiguous high-severity anomaly)
                 │
                 ├── Medium/High Severity AND Area 3.0% - 15.0% ─► ACTION: GRIND (Rule 3)
                 │   Class in [Scratches, Patches, Scale, Inclusion]
                 │
                 └── Low Severity OR Area < 3.0% ────────────────► ACTION: INSPECT (Rule 4)
```

---

## Repository & Codebase Structure

```
JSL-SPARK/
├── .env.example                               # Template for environment variables (GROQ_API_KEY)
├── packages.txt                               # System dependencies for Streamlit Cloud (libgl1, etc.)
├── requirements.txt                           # Python dependencies (headless OpenCV, PyTorch, ChromaDB)
├── README.md                                  # Comprehensive platform documentation
│
├── data/
│   └── NEU-DET-final/                         # Benchmark steel surface dataset
│       ├── data.yaml                          # YOLO dataset configuration
│       └── images/val/                        # 288 validation images across all 6 classes (48 each)
│
├── models/
│   ├── best.pt                                # Fine-tuned YOLOv8 Nano model weights
│   ├── calibration_params.json                # Fitted Temperature Scaling parameter (T=2.397)
│   └── ood_svm.pkl                            # Trained One-Class SVM anomaly model
│
├── src/
│   ├── app.py                                 # Industrial Streamlit multi-tab web application
│   ├── detect.py                              # YOLOv8 inference wrapper & bounding box extractor
│   ├── characterize.py                        # Morphology, centroid location, area %, severity scoring
│   ├── calibrate.py                           # Post-hoc Temperature Scaling calibrator & ECE calculator
│   ├── anomaly.py                             # One-Class SVM feature extractor & divergence detector
│   ├── gradcam.py                             # Convolutional activation gradients utility module
│   ├── process_correlate.py                   # Mill process telemetry correlation engine
│   ├── decide.py                              # Rule-based operational decision matrix
│   ├── feedback.py                            # Operator confirmation & correction re-indexing loop
│   ├── rag_assistant.py                       # Conversational metallurgical RAG Q&A engine
│   │
│   └── kb/
│       ├── steel_surface_metallurgy_rag_kb_final.json  # 48 curated metallurgical passages & 11 sources
│       ├── passages.json                      # Backward-compatible formatted passage store
│       ├── synthetic_process_data.json        # Simulated mill telemetry batches per defect class
│       ├── build_index.py                     # ChromaDB embedding & indexing pipeline
│       ├── retrieve.py                        # Cosine similarity vector retrieval with class filters
│       └── chroma_db/                         # Persistent ChromaDB vector index directory
│
└── tests/
    ├── test_phase3.py                         # Vision detection & characterization verification
    ├── test_phase4.py                         # Metallurgical vector RAG & diagnostic policy tests
    ├── test_phase5.py                         # Operational decision engine triage tests
    └── test_end_to_end.py                     # Full 4-specimen integration test pipeline
```

---

## Installation & Local Quickstart

### 1. Prerequisites
- **Python:** Version 3.11 or 3.12
- **Operating System:** Windows, Linux, or macOS
- **Hardware:** CPU supported; CUDA-compatible GPU recommended for accelerated inference

### 2. Clone the Repository
```bash
git clone https://github.com/Aman-k-s/JSL-SPARK.git
cd JSL-SPARK
```

### 3. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure API Credentials
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
```
*(Get your free API key at [console.groq.com](https://console.groq.com)).*

### 6. Build or Verify Vector Index
```bash
python src/kb/build_index.py
```

### 7. Run the Application
```bash
streamlit run src/app.py
```
Open your browser at `http://localhost:8501`.

---

## Streamlit Cloud Deployment Guide

This repository is pre-configured for zero-friction continuous deployment on **Streamlit Cloud**:

1. **Repository Link:** Connect your GitHub account and select `Aman-k-s/JSL-SPARK`.
2. **Main File Path:** Set to `src/app.py`.
3. **Headless Linux Support:** The repository includes `packages.txt` (`libgl1`, `libglib2.0-0`) and `requirements.txt` (`opencv-python-headless`) to ensure seamless execution on cloud containers without display servers.
4. **Secrets Configuration:**
   In your Streamlit Cloud Dashboard under **Settings → Secrets**, add your Groq key:
   ```toml
   GROQ_API_KEY = "gsk_your_actual_groq_api_key_here"
   ```

---

## Test Suite Execution

Validate all platform modules with the automated test suite:

```bash
# Run Vision & Characterization Tests
python tests/test_phase3.py

# Run Vector RAG & Groq Synthesis Tests
python tests/test_phase4.py

# Run Operational Decision Engine Tests
python tests/test_phase5.py

# Run Complete End-to-End Test Suite across Real Specimens
python tests/test_end_to_end.py
```

When all tests pass, the output displays:
```
=====================================================================================
ALL 4 REAL TEST IMAGES PASSED END-TO-END PIPELINE VERIFICATION!
=====================================================================================
```

---

## Authors & Attribution

- **Platform Architecture & Development:** [Aman-k-s](https://github.com/Aman-k-s) (`aks211531@gmail.com`)
- **Domain Literature & Failure Mode Evidence:** Curated metallurgical failure passages and technical sources (`S1`–`S11`).
- **Benchmark Dataset:** NEU-DET (Northeastern University Surface Defect Database).
