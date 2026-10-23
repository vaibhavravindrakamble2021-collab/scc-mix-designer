import streamlit as st
import joblib
import pandas as pd
from fpdf import FPDF

# Page Configuration
st.set_page_config(page_title="SCC Mix Designer", page_icon="🏗️", layout="centered")

# Load the trained model and scaler
model = joblib.load("scc_model.pkl")
scaler = joblib.load("scc_scaler.pkl")

# Header with Logo and College Name side-by-side
# Added vertical_alignment="center" to keep them perfectly leveled
col1, col2 = st.columns([1, 4], vertical_alignment="center")

with col1:
    # College Logo
    st.image("d_y_patil_college_of_engineering__technology_kasaba_bawada_logo.jpeg", width=130)

with col2:
    # Changed margin-top to 0px so the text moves up to match the image height
    st.markdown(
        """
        <div style="margin-top: 0px;">
            <h2 style="margin-bottom: 5px; padding: 0;">D. Y. Patil College of Engineering & Technology</h2>
            <hr style="margin: 0; border: none; border-top: 2px solid #888;">
            <h4 style="margin-top: 5px; padding: 0; font-weight: normal; color: #cccccc;">Kasaba Bawada, Kolhapur 416006</h4>
        </div>
        """,
        unsafe_allow_html=True
    )

# Title and Academic Details
st.title("🏗️ Artificial Neural Network for SCC Mix Design")
st.markdown("### Final Year B.Tech Civil Engineering Project")
st.markdown("**Developed by:** Vaibhav Ravindra Kamble")
st.markdown("**Institution:** D. Y. Patil College of Engineering and Technology, Kolhapur")
st.divider()

st.write("This intelligent calculator uses machine learning to predict exact material quantities for Self-Compacting Concrete. Enter your target concrete grade below.")

# Input field
target_strength = st.number_input("Target Compressive Strength (MPa) [e.g., 30 for M30]", min_value=15.0, max_value=90.0, value=30.0, step=1.0)

# Function to generate PDF
def create_pdf(grade, ratio, df):
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Self-Compacting Concrete Mix Design Report", ln=True, align='C')
    pdf.ln(5)
    
    # Project Details
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 8, "Project: Final Year B.Tech Civil Engineering", ln=True, align='C')
    pdf.cell(0, 8, "Developed by: Vaibhav Ravindra Kamble", ln=True, align='C')
    pdf.cell(0, 8, "Institution: D. Y. Patil College of Engineering and Technology, Kolhapur", ln=True, align='C')
    pdf.ln(10)
    
    # Grade and Ratio
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, f"Target Concrete Grade: M{int(grade)}", ln=True)
    pdf.cell(0, 10, f"Estimated Mix Ratio (Binder : FA : CA) -> {ratio}", ln=True)
    pdf.ln(5)
    
    # Table Header
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(90, 10, "Material Parameter", border=1)
    pdf.cell(70, 10, "Predicted Quantity", border=1, ln=True)
    
    # Table Data
    pdf.set_font("Arial", '', 12)
    for i in range(len(df)):
        pdf.cell(90, 10, str(df.iloc[i, 0]), border=1)
        pdf.cell(70, 10, str(df.iloc[i, 1]), border=1, ln=True)
        
    # Standard IS Code note
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, "Note: Predictions are based on ANN model adhering to IS 10262:2019 principles.", ln=True)
    
    # Output PDF as bytes for download
    return pdf.output(dest='S').encode('latin-1')

# Generate Button
if st.button("Generate Mix Design", type="primary"):
    input_scaled = scaler.transform([[target_strength]])
    prediction = model.predict(input_scaled)[0]
    
    total_binder = prediction[0]
    fly_ash = prediction[1]
    w_b_ratio = prediction[2]
    fine_agg = prediction[3]
    coarse_agg = prediction[4]
    superplast = prediction[5]
    
    fa_ratio = fine_agg / total_binder
    ca_ratio = coarse_agg / total_binder
    mix_ratio = f"1 : {fa_ratio:.2f} : {ca_ratio:.2f}"
    
    st.success(f"Successfully generated mix proportions for M{int(target_strength)} Grade!")
    st.markdown(f"#### **Estimated Mix Ratio (Binder : FA : CA) &rarr; {mix_ratio}**")
    
    results_df = pd.DataFrame({
        "Material Parameter": ["Total Binder", "Fly Ash Replacement", "Water/Binder Ratio", "Fine Aggregate", "Coarse Aggregate", "Superplasticizer"],
        "Predicted Quantity": [f"{total_binder:.2f} kg/m³", f"{fly_ash:.2f} %", f"{w_b_ratio:.2f}", f"{fine_agg:.2f} kg/m³", f"{coarse_agg:.2f} kg/m³", f"{superplast:.2f} kg/m³"]
    })
    st.table(results_df)
    
    # Download Section layout
    st.markdown("### 📥 Download Reports")
    col1, col2 = st.columns(2)
    
    with col1:
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download Excel (CSV)",
            data=csv,
            file_name=f"SCC_Mix_Design_M{int(target_strength)}.csv",
            mime="text/csv",
        )
        
    with col2:
        pdf_bytes = create_pdf(target_strength, mix_ratio, results_df)
        st.download_button(
            label="📑 Download PDF Report",
            data=pdf_bytes,
            file_name=f"SCC_Mix_Design_M{int(target_strength)}.pdf",
            mime="application/pdf",
        )

st.divider()
st.info("**Dataset Reference:** Trained on a standard Mendeley experimental dataset containing 114 laboratory mix designs of Self-Compacting Concrete, adhering to IS 10262:2019 proportioning principles.")