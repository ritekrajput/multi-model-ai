import streamlit as st
import requests

# -----------------------
# CONFIG
# -----------------------
BACKEND_URL = "http://127.0.0.1:8000/assessments/create"

st.set_page_config(
    page_title="Depression Assessment",
    layout="centered"
)

# -----------------------
# UI HEADER
# -----------------------
st.title("🧠 Depression Severity Assessment")
st.caption("AI-assisted screening tool (academic & research use only)")

st.markdown("---")

# -----------------------
# USER INPUT
# -----------------------
st.subheader("👤 Patient Information")

user_id = st.text_input(
    "Patient ID",
    placeholder="e.g. patient_001"
)

patient_text = st.text_area(
    "How have you been feeling recently?",
    placeholder="Describe thoughts, mood, sleep, motivation, etc."
)

st.markdown("---")

# -----------------------
# RELATIVE INPUT
# -----------------------
st.subheader("👨‍👩‍👧 Relative / Caregiver Input (Optional)")

relationship = st.selectbox(
    "Relationship",
    ["", "Parent", "Sibling", "Partner", "Friend"]
)

relative_text = st.text_area(
    "Observed behavioral changes",
    placeholder="Any withdrawal, mood changes, routine disruption?"
)

st.markdown("---")

# -----------------------
# SUBMIT BUTTON
# -----------------------
if st.button("🧪 Run Assessment"):
    if not user_id or not patient_text:
        st.error("Patient ID and patient input are required.")
    else:
        payload = {
            "user_id": user_id,
            "patient_text": patient_text,
            "relative_text": relative_text if relative_text else None,
            "relationship": relationship if relationship else None
        }

        with st.spinner("Running AI inference..."):
            try:
                response = requests.post(BACKEND_URL, json=payload)

                if response.status_code != 200:
                    st.error("Backend error occurred.")
                    st.stop()

                result = response.json()

            except Exception as e:
                st.error(f"Could not connect to backend: {e}")
                st.stop()

        # -----------------------
        # RESULTS DISPLAY
        # -----------------------
        st.success("Assessment completed")

        st.subheader("📊 Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Severity Score",
                value=f"{result['severity_score']:.2f} / 10"
            )

        with col2:
            st.metric(
                label="Severity Level",
                value=result["severity_level"]
            )

        st.markdown("**Recommended Action**")
        st.info(result["recommendation"])

        if result["risk_flags"]:
            st.markdown("**⚠️ Risk Flags Detected**")
            for flag in result["risk_flags"]:
                st.warning(flag)
        else:
            st.success("No high-risk indicators detected.")

# -----------------------
# FOOTER
# -----------------------
st.markdown("---")
st.caption(
    "⚠️ Disclaimer: This tool is for academic and research purposes only. "
    "It does not replace professional medical diagnosis."
)
