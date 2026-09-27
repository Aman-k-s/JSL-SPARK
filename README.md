# JSL-SPARK: Hot-Rolled Steel Defect Vision & Root-Cause Platform

Automated end-to-end industrial inspection platform for hot-rolled steel surface flaws. Combines fine-tuned YOLOv8 object detection, morphological fingerprinting, ChromaDB vector retrieval over metallurgical failure literature, deterministic operational decisioning, Grad-CAM visual explainability, One-Class SVM anomaly detection, and an interactive domain RAG assistant.

---

## Architecture Overview

```
                      +-----------------------------+
                      |   Steel Surface Specimen    |
                      +--------------+--------------+
                                     |
                                     v
                       [ YOLOv8n Detection Model ]
                         (models/best.pt weights)
                                     |
                +--------------------+--------------------+
                |                                         |
                v                                         v
   [ Morphological Fingerprint ]               [ Anomaly Detector ]
   (aspect ratio, centroid, area)              (One-Class SVM on SPPF)
                |                                         |
                v                                         v
   [ ChromaDB Vector RAG ]                     [ Out-of-Distribution Alert ]
   (36+ metallurgical passages)
                |
                v
   [ Grounded LLM Diagnosis ]
   (Groq LLM + strict citation)
                |
                v
   [ Operational Decision Engine ]
   (Inspect / Grind / Escalate)
```

---

## Module Structure & Team Ownership

- **`src/detect.py`**: Ultralytics YOLOv8 wrapper on `models/best.pt` extracting bounding boxes, classes, and confidence scores.
- **`src/characterize.py`**: Computes morphology (elongated vs compact), spatial location (edge vs center via 15% outer margin), affected area %, calibrated severity score, and surface pattern (isolated vs repetitive).
- **`src/calibrate.py`**: Post-hoc Temperature Scaling confidence calibrator fitted on validation set detections (35.5% ECE reduction).
- **`src/kb/passages.json`**: 48 domain-researched metallurgical passages across 6 NEU-DET defect classes (crazing, inclusion, patches, pitted surface, rolled-in scale, scratches) backed by 11 technical sources.
- **`src/kb/build_index.py`**: Embeds passages with `sentence-transformers` (`all-MiniLM-L6-v2`) and builds ChromaDB vector index.
- **`src/kb/retrieve.py`**: Vector similarity retrieval returning cosine similarity scores and metadata.
- **`src/diagnose.py`**: Grounded root-cause diagnosis using Groq LLM with strict passage citations and diagnostic investigation actions.
- **`src/decide.py`**: Rule-based decision layer determining operational disposition (`Inspect` / `Grind` / `Escalate`).
- **`src/gradcam.py`**: Activation heatmap overlay on the YOLOv8 backbone SPPF layer for visual explainability.
- **`src/anomaly.py`**: One-Class SVM on pooled YOLOv8 backbone features detecting out-of-distribution flaws.
- **`src/process_correlate.py`**: Correlates hot strip mill process telemetry (reheat temp, roll speed, chemistry) with defects.
- **`src/feedback.py`**: Active feedback loop recording operator confirmations and domain corrections with live ChromaDB re-indexing.
- **`src/rag_assistant.py`**: Interactive RAG assistant answering technical questions grounded in the knowledge base and active inspection context.
- **`src/app.py`**: Industrial Streamlit application integrating all stages.

---

## Installation & Quickstart

### 1. Requirements
- Python 3.11 or 3.12
- PyTorch & Torchvision
- Ultralytics YOLO
- ChromaDB & Sentence-Transformers
- Groq Python SDK
- Streamlit & OpenCV

### 2. Environment Configuration
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Launch the Application
```bash
streamlit run src/app.py
```
Or execute the Windows launcher:
```powershell
.\run_app.bat
```

### 4. Run Test Suites
```bash
# Phase 3: Detection & Characterization
python tests/test_phase3.py

# Phase 4: Grounded Diagnosis (6 classes)
python tests/test_phase4.py

# Phase 5: Decision Matrix
python tests/test_phase5.py

# End-to-End Pipeline
python tests/test_end_to_end.py
```

---

## License & Attribution
Developed for the JSL-SPARK Hot-Rolled Steel Defect Detection & Root-Cause Analysis Initiative.
Dataset: NEU-DET (Northeastern University Surface Defect Database).
