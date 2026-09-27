"""
Module: src/app.py
Description: Enterprise industrial surface defect inspection and metallurgical root-cause platform.
             Features front-page specimen upload, automated YOLOv8 inference, morphological fingerprinting,
             ChromaDB vector RAG diagnosis based on 48 curated metallurgical passages and 11 technical sources,
             operational decisioning, Grad-CAM explainability, and an interactive domain assistant.
# OWNER: Plant Quality Systems Engineering
"""

import os
import sys
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image
import streamlit as st

# Configure sys.path so imports work seamlessly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detect import detect
from src.characterize import characterize, detect_pattern
from src.diagnose import diagnose
from src.decide import decide
from src.calibrate import calibrate_confidence
from src.gradcam import generate_gradcam
from src.anomaly import evaluate_anomaly
from src.process_correlate import correlate_process_parameters
from src.feedback import record_confirmation, record_correction
from src.rag_assistant import answer_question

# Page configuration
st.set_page_config(
    page_title="Jindal Stainless | Surface Quality Inspection & Root-Cause Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise Industrial Styling
st.markdown("""
<style>
    /* Global layout & typography */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #0F172A;
    }
    .header-title {
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #0F172A;
        margin-bottom: 0.15rem;
        text-transform: uppercase;
    }
    .header-subtitle {
        font-size: 0.88rem;
        color: #64748B;
        font-weight: 400;
        margin-bottom: 1.25rem;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 0.75rem;
    }
    .upload-card {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 4px;
        padding: 16px 20px;
        margin-bottom: 18px;
    }
    .panel-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 4px;
        padding: 16px;
        margin-bottom: 14px;
    }
    .panel-card-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .status-badge {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 4px 10px;
        border-radius: 3px;
    }
    .status-escalate {
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
    }
    .status-grind {
        background-color: #FFFBEB;
        color: #92400E;
        border: 1px solid #FCD34D;
    }
    .status-inspect {
        background-color: #F0FDF4;
        color: #166534;
        border: 1px solid #86EFAC;
    }
    .status-anomaly {
        background-color: #FAF5FF;
        color: #6B21A8;
        border: 1px solid #D8B4FE;
    }
    .investigation-box {
        background-color: #F8FAFC;
        border-left: 3px solid #0284C7;
        padding: 10px 14px;
        margin-top: 10px;
        font-size: 0.88rem;
        color: #0F172A;
        line-height: 1.5;
    }
    .citation-tag {
        font-family: Consolas, monospace;
        font-size: 0.78rem;
        background-color: #F1F5F9;
        color: #0F172A;
        padding: 2px 6px;
        border: 1px solid #CBD5E1;
        border-radius: 3px;
    }
    .disclaimer-box {
        font-size: 0.78rem;
        color: #475569;
        background-color: #F8FAFC;
        border-left: 3px solid #64748B;
        padding: 6px 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


def draw_detections(image_np: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes and class labels with high-contrast industrial annotation."""
    annotated = image_np.copy()
    
    color_map = {
        "crazing": (220, 38, 38),       # Red
        "inclusion": (234, 88, 12),     # Orange
        "patches": (202, 138, 4),       # Amber
        "pitted_surface": (13, 148, 136),# Teal
        "rolled_in_scale": (37, 99, 235),# Blue
        "rolled-in_scale": (37, 99, 235),# Blue
        "scratches": (147, 51, 234)     # Purple
    }

    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        cls_name = d["class"]
        conf = d["confidence"]
        color = color_map.get(cls_name, (75, 85, 99))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label_text = f"{cls_name.upper()} [{conf:.2f}]"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        y_text = max(y1 - 4, th + 4)
        cv2.rectangle(annotated, (x1, y_text - th - 3), (x1 + tw + 4, y_text + 3), color, -1)
        cv2.putText(annotated, label_text, (x1 + 2, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    return annotated


# Sidebar: Parameters & Knowledge Base Metadata
st.sidebar.markdown("### System Parameters")
st.sidebar.caption("Hot Strip Mill Quality Control Node")

conf_threshold = st.sidebar.slider("Detection Confidence Cutoff", min_value=0.10, max_value=0.90, value=0.25, step=0.05)
enable_gradcam = st.sidebar.checkbox("Compute Grad-CAM Activation Heatmap", value=False)
anomaly_threshold = st.sidebar.slider("Out-of-Distribution Sensitivity", min_value=0.40, max_value=0.90, value=0.65, step=0.05)


# Main Header
st.markdown('<div class="header-title">Hot-Rolled Steel Surface Quality & Root-Cause Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">JSL Vision Diagnostics | Automated YOLOv8 Detection | Morphological Fingerprinting | Metallurgical Domain RAG (48 Curated Passages, 11 Technical Sources)</div>', unsafe_allow_html=True)

# Main Navigation Tabs
tab_inspection, tab_assistant = st.tabs(["Automated Surface Inspection", "Metallurgical RAG Assistant"])

# Cross-tab context session state
if "active_fingerprint" not in st.session_state:
    st.session_state.active_fingerprint = None
if "active_diagnosis" not in st.session_state:
    st.session_state.active_diagnosis = None
if "rag_chat_history" not in st.session_state:
    st.session_state.rag_chat_history = [
        {
            "role": "assistant",
            "content": "Metallurgical retrieval assistant initialized with curated 48-passage domain knowledge base (11 technical sources). Submit an inquiry regarding defect mechanisms, roll wear, descaling headers, or reference the active surface inspection."
        }
    ]

# TAB 1: Inspection View
with tab_inspection:
    # Front-and-Center Image Upload & Specimen Selection Card
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-card-title">Specimen Input & Image Upload</div>', unsafe_allow_html=True)
    
    val_dir = PROJECT_ROOT / "data" / "NEU-DET-final" / "images" / "val"

    up_col1, up_col2 = st.columns([1, 1.4])
    
    with up_col1:
        input_source_mode = st.radio(
            "Input Selection Mode:",
            ["Upload Specimen Image", "Select From Validation Gallery"],
            horizontal=False
        )

    selected_image_path = None
    with up_col2:
        if input_source_mode == "Upload Specimen Image":
            uploaded_file = st.file_uploader(
                "Upload Steel Surface Specimen (JPG / JPEG / PNG):",
                type=["jpg", "jpeg", "png"],
                help="Drag and drop or browse for a high-resolution steel strip surface image."
            )
            if uploaded_file is not None:
                temp_dir = PROJECT_ROOT / "scratch"
                os.makedirs(temp_dir, exist_ok=True)
                selected_image_path = temp_dir / uploaded_file.name
                with open(selected_image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
        else:
            class_options = {
                "All Defect Classes": None,
                "Crazing": "crazing",
                "Inclusion": "inclusion",
                "Patches": "patches",
                "Pitted Surface": "pitted_surface",
                "Rolled-in Scale": "rolled-in_scale",
                "Scratches": "scratches",
            }
            
            sub_c1, sub_c2 = st.columns([1, 1.2])
            with sub_c1:
                selected_class_label = st.selectbox(
                    "Filter Defect Class:",
                    list(class_options.keys()),
                    index=0,
                    help="Filter validation specimens across all 6 defect classes."
                )

            target_prefix = class_options[selected_class_label]
            if target_prefix:
                filtered_samples = [f.name for f in sorted(val_dir.glob(f"{target_prefix}_*.jpg"))]
            else:
                # Include specimens from all 6 classes evenly
                all_files_by_class = []
                for pfx in ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]:
                    all_files_by_class.extend([f.name for f in sorted(val_dir.glob(f"{pfx}_*.jpg"))])
                filtered_samples = all_files_by_class

            with sub_c2:
                chosen_sample = st.selectbox(
                    f"Select Specimen ({len(filtered_samples)} available):",
                    filtered_samples,
                    index=0 if filtered_samples else None,
                    help="Select a specimen from the validation set to run vision and metallurgical diagnosis."
                )

            if chosen_sample:
                selected_image_path = val_dir / chosen_sample

    st.markdown('</div>', unsafe_allow_html=True)

    if not selected_image_path or not selected_image_path.exists():
        st.info("Upload an image above or select a specimen from the validation gallery to begin automated analysis.")
    else:
        img_bgr = cv2.imread(str(selected_image_path))
        if img_bgr is None:
            st.error(f"Failed to decode specimen image: {selected_image_path}")
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]

            # Execute pipeline
            with st.spinner("Executing inference and vector retrieval over 48 curated passages..."):
                raw_detections = detect(str(selected_image_path), conf_threshold=conf_threshold)
                pattern = detect_pattern(raw_detections)
                fingerprints = []
                for d in raw_detections:
                    fp = characterize(d, (h, w))
                    fp["pattern"] = pattern
                    fingerprints.append(fp)

                max_conf = max([d["confidence"] for d in raw_detections], default=0.0)
                anomaly_result = evaluate_anomaly(str(selected_image_path), yolo_confidence=max_conf)

            # Anomaly alert banner if triggered
            if anomaly_result["is_anomaly"]:
                st.markdown(f"""
                <div style="background-color: #FAF5FF; border: 1px solid #D8B4FE; border-left: 4px solid #9333EA; padding: 12px 16px; margin-bottom: 16px;">
                    <div class="status-badge status-anomaly" style="margin-bottom: 6px;">Alert: Out-of-Distribution Defect Candidate</div>
                    <div style="color: #581C87; font-size: 0.88rem;">
                        Classification confidence is depressed ({max_conf:.2f}) while One-Class SVM feature divergence is elevated 
                        (anomaly score: {anomaly_result['anomaly_score']:.3f} >= threshold: {anomaly_result['threshold']:.2f}). 
                        Specimen does not conform to nominal NEU-DET defect distributions and is flagged for metallurgist verification.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            col_img, col_metrics = st.columns([1.1, 1.4])

            with col_img:
                st.markdown('<div class="panel-card-title">Specimen Imagery & Spatial Annotations</div>', unsafe_allow_html=True)
                annotated_img = draw_detections(img_rgb, raw_detections)
                st.image(annotated_img, caption=f"Specimen: {selected_image_path.name} | Resolution: {w}x{h} | Detections: {len(raw_detections)}", use_container_width=True)

                if enable_gradcam:
                    with st.spinner("Computing convolutional activation gradients..."):
                        gradcam_overlay, _ = generate_gradcam(str(selected_image_path))
                        st.image(gradcam_overlay, caption="Grad-CAM Activation Heatmap (Backbone SPPF Layer)", use_container_width=True)

            with col_metrics:
                st.markdown('<div class="panel-card-title">Defect Characterization & Metallurgical Diagnosis</div>', unsafe_allow_html=True)

                if not fingerprints:
                    st.success("No surface anomalies detected above threshold. Surface meets nominal acceptance criteria.")
                    st.session_state.active_fingerprint = None
                    st.session_state.active_diagnosis = None
                else:
                    if len(fingerprints) > 1:
                        det_labels = [f"Instance #{i+1}: {fp['class'].upper()} (Conf: {fp['confidence']:.2f}, Severity: {fp['severity']})" for i, fp in enumerate(fingerprints)]
                        selected_idx = st.selectbox("Active Defect Instance:", range(len(fingerprints)), format_func=lambda i: det_labels[i])
                    else:
                        selected_idx = 0

                    target_fp = fingerprints[selected_idx]
                    st.session_state.active_fingerprint = target_fp

                    # Metric Summary Bar
                    m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                    m_c1.metric("Class", target_fp["class"].replace("_", " ").title())
                    m_c2.metric("Raw Confidence", f"{target_fp['confidence']:.2f}", delta=f"Calibrated: {target_fp['calibrated_confidence']:.2f}")
                    m_c3.metric("Severity Level", target_fp["severity"])
                    m_c4.metric("Affected Area", f"{target_fp['affected_area_pct']:.2f}%")

                    # Grounded Diagnosis & Operational Decision Engine
                    with st.spinner("Querying curated metallurgical knowledge base & evaluating disposition..."):
                        diag_result = diagnose(target_fp, k=3)
                        st.session_state.active_diagnosis = diag_result
                        decision = decide(target_fp, diag_result)

                    action = decision["action"]
                    badge_class = "status-inspect"
                    border_color = "#2563EB"
                    if action == "Escalate":
                        badge_class = "status-escalate"
                        border_color = "#DC2626"
                    elif action == "Grind":
                        badge_class = "status-grind"
                        border_color = "#D97706"

                    # Recommended Operational Disposition placed at the top of results
                    st.markdown(f"""
                    <div class="panel-card" style="margin-top: 10px; border-left: 4px solid {border_color};">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <div class="panel-card-title" style="margin: 0; font-size: 1.05rem;">Recommended Operational Disposition</div>
                            <span class="status-badge {badge_class}">ACTION: {action.upper()}</span>
                        </div>
                        <div style="font-size: 0.92rem; color: #1E293B; line-height: 1.5; font-weight: 500;">
                            {decision['rationale']}
                        </div>
                        <div style="font-size: 0.78rem; color: #64748B; margin-top: 6px;">
                            Triggered Protocol: <code>{decision['rule_triggered']}</code>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander("Morphological Metrics Summary", expanded=False):
                        st.json({
                            "classification": target_fp["class"],
                            "bounding_box": target_fp["bbox"],
                            "raw_confidence": target_fp["confidence"],
                            "calibrated_confidence": target_fp["calibrated_confidence"],
                            "aspect_ratio": target_fp["aspect_ratio"],
                            "morphology": target_fp["morphology"],
                            "strip_location": target_fp["location"],
                            "normalized_centroid": target_fp["normalized_centroid"],
                            "affected_area_percent": target_fp["affected_area_pct"],
                            "distribution_pattern": target_fp["pattern"],
                            "computed_severity_score": target_fp["severity_score"]
                        })

                    # Display Diagnostic Output Contract
                    st.markdown(f"""
                    <div class="panel-card" style="margin-top: 10px;">
                        <div class="panel-card-title">Metallurgical Probable-Origin Hypothesis</div>
                        <div style="font-size: 0.92rem; color: #1E293B; line-height: 1.55; margin-bottom: 8px;">
                            {diag_result['probable_origin_hypothesis']}
                        </div>
                        <div class="investigation-box">
                            <b>Recommended Investigation Action:</b><br/>
                            {diag_result['recommended_investigation']}
                        </div>
                        <div style="font-size: 0.8rem; color: #64748B; margin-top: 10px;">
                            Cited KB Passage: <span class="citation-tag">{diag_result['cited_passage_id']}</span> | 
                            Evidence Level: <code>{diag_result['evidence_level'].upper()}</code> | 
                            Source IDs: <code>{diag_result['source_ids']}</code> | 
                            Retrieval Similarity: <code>{diag_result['confidence']:.4f}</code>
                        </div>
                        <div style="font-size: 0.76rem; color: #94A3B8; margin-top: 4px;">
                            References: {diag_result['source_citations']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Process Correlation
                    with st.expander("Process Telemetry Correlation", expanded=False):
                        st.markdown('<div class="disclaimer-box">NOTICE: Telemetry values represent simulated process variables for evaluation purposes.</div>', unsafe_allow_html=True)
                        corr_data = correlate_process_parameters(target_fp["class"])
                        t_col1, t_col2 = st.columns(2)
                        with t_col1:
                            st.write(f"**Coil Batch ID:** `{corr_data['batch_id']}`")
                            st.write(f"**Steel Grade:** `{corr_data['steel_grade']}`")
                            for k, v in list(corr_data["telemetry"].items())[:3]:
                                st.write(f"- {k}: {v}")
                        with t_col2:
                            for k, v in list(corr_data["telemetry"].items())[3:]:
                                st.write(f"- {k}: {v}")
                        st.markdown(f"**Metallurgical Kinetic Mechanism:** {corr_data['correlation_mechanism']}")
                        if corr_data["anomalous_parameters"]:
                            st.warning("Flagged Parameter Deviations: " + "; ".join(corr_data["anomalous_parameters"]))

                    # Active Feedback
                    st.markdown('<div class="panel-card-title" style="margin-top: 14px;">Domain Feedback & Knowledge Validation</div>', unsafe_allow_html=True)
                    f_col1, f_col2 = st.columns(2)
                    with f_col1:
                        if st.button("Confirm Diagnosis Assessment", use_container_width=True):
                            res = record_confirmation(target_fp, diag_result)
                            st.success(f"Assessment confirmed and recorded under Event ID: {res['event_id']}")
                    with f_col2:
                        with st.popover("Submit Technical Correction", use_container_width=True):
                            st.markdown("**Domain Expert Correction Form**")
                            corr_class = st.selectbox("Validated Defect Classification:", ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"], index=["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"].index(target_fp["class"]) if target_fp["class"] in ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"] else 0)
                            corr_text = st.text_area("Corrective Metallurgical Explanation:", placeholder="Specify validated root cause, mill contact point, or thermal mechanism...")
                            if st.button("Commit to Knowledge Base", type="primary"):
                                if corr_text.strip():
                                    with st.spinner("Appending passage and re-indexing ChromaDB..."):
                                        c_res = record_correction(corr_class, corr_text, target_fp, diag_result)
                                        st.success(f"{c_res['message']} (Active Knowledge Base Passages: {c_res['total_passages']})")
                                else:
                                    st.warning("Correction explanation text cannot be empty.")


# TAB 2: Metallurgical RAG Assistant
with tab_assistant:
    st.markdown('<div class="panel-card-title">Metallurgical Knowledge Base Retrieval Assistant</div>', unsafe_allow_html=True)
    st.caption("Grounded retrieval and diagnostic hypothesis generation over 48 curated metallurgical passages backed by 11 technical sources.")

    if st.session_state.active_fingerprint:
        active_cls = st.session_state.active_fingerprint.get("class").upper()
        st.info(f"Active Specimen Context Attached: Defect Class [{active_cls}], Severity [{st.session_state.active_fingerprint.get('severity')}]. Questions referencing this defect will be contextualized automatically.")

    # Suggested Prompts
    st.markdown("**Standard Inquiries:**")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    selected_prompt = None

    with p_col1:
        if st.button("Descaling & rolled-in scale", use_container_width=True):
            selected_prompt = "What causes rolled-in scale, what descaling pressure is required, and what investigation is recommended?"
    with p_col2:
        if st.button("Side guide wear & scratches", use_container_width=True):
            selected_prompt = "How do we prevent scratches caused by side guide wear and alignment?"
    with p_col3:
        if st.button("Copper tramp elements & crazing", use_container_width=True):
            selected_prompt = "Why do residual copper and tramp elements cause hot shortness and crazing?"
    with p_col4:
        if st.button("Contextual analysis of active specimen", use_container_width=True):
            selected_prompt = "Explain the probable root causes of the defect currently detected in the inspection tab and recommend plant investigation actions."

    # Chat Transcript Display
    for msg in st.session_state.rag_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "citations" in msg and msg["citations"]:
                with st.expander("Retrieved Evidence Passages & Sources", expanded=False):
                    for p in msg["citations"]:
                        st.markdown(f"**Passage [{p['id']}]** | Class: `{p['defect_class']}` | Similarity Score: `{p['score']:.3f}`")
                        st.markdown(f"> {p['text']}")
                        st.caption(f"Source IDs: {p.get('source_ids')} | Evidence Level: {p.get('evidence_level')} | References: {p.get('source')}")

    # User Input
    user_query = st.chat_input("Enter metallurgical or rolling process inquiry...")
    active_query = selected_prompt or user_query

    if active_query:
        st.session_state.rag_chat_history.append({"role": "user", "content": active_query})
        with st.chat_message("user"):
            st.markdown(active_query)

        with st.chat_message("assistant"):
            with st.spinner("Querying knowledge base index and synthesizing grounded response..."):
                rag_res = answer_question(
                    user_query=active_query,
                    current_fingerprint=st.session_state.active_fingerprint,
                    current_diagnosis=st.session_state.active_diagnosis,
                    k=4
                )
                st.markdown(rag_res["answer"])
                if rag_res["cited_passages"]:
                    with st.expander("Retrieved Evidence Passages & Sources", expanded=False):
                        for p in rag_res["cited_passages"]:
                            st.markdown(f"**Passage [{p['id']}]** | Class: `{p['defect_class']}` | Similarity Score: `{p['score']:.3f}`")
                            st.markdown(f"> {p['text']}")
                            st.caption(f"Source IDs: {p.get('source_ids')} | Evidence Level: {p.get('evidence_level')} | References: {p.get('source')}")

        st.session_state.rag_chat_history.append({
            "role": "assistant",
            "content": rag_res["answer"],
            "citations": rag_res["cited_passages"]
        })
        st.rerun()
