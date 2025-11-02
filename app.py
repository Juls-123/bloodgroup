import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import time
from io import BytesIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# Page configuration
st.set_page_config(
    page_title="Blood Group & Genotype Predictor | Clinical AI Platform",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/blood-predictor',
        'Report a bug': "https://github.com/yourusername/blood-predictor/issues",
        'About': "# Blood Group & Genotype Predictor\nVersion 2.0\n\nProfessional AI-powered clinical diagnostic tool"
    }
)

# Custom CSS for professional industry-standard design
# Cache buster: v2.1
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@400;500;600;700;800&display=swap');
    
    /* Global Styles - v2.1 */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main {
        padding: 1rem 2rem;
        background: #f8fafc;
    }
    
    /* Header Styles - Soft Gradient with Depth */
    .header-container {
        background: linear-gradient(135deg, #102a43 0%, #243b55 50%, #1e3a8a 100%);
        padding: 3rem 2rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(16, 42, 67, 0.25);
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    
    .header-title {
        color: #ffffff !important;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-align: center;
        letter-spacing: -0.5px;
        position: relative;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    .header-subtitle {
        color: #cbd5e1;
        font-size: 1.2rem;
        font-weight: 400;
        text-align: center;
        margin-bottom: 0;
        position: relative;
    }
    
    /* Card Styles with Micro Animations */
    .card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
        transition: all 0.2s ease-in-out;
        animation: fadeIn 0.3s ease-in;
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .card:hover {
        transform: scale(1.01);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
    }
    
    .card-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        text-shadow: 0 0 2px rgba(0,0,0,0.05);
    }
    
    /* Card descriptions on light backgrounds */
    .card p {
        color: #475569;
    }
    
    /* Button Styles */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        border-radius: 12px;
        padding: 1rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(30, 58, 138, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #1e40af 0%, #2563eb 100%);
        box-shadow: 0 6px 25px rgba(30, 58, 138, 0.5);
        transform: translateY(-2px);
    }
    
    .stButton>button:active {
        transform: translateY(0);
    }
    
    /* Prediction Result Cards */
    .prediction-box {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(5, 150, 105, 0.25);
        position: relative;
        overflow: hidden;
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: scale(0.95);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    .prediction-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, transparent 100%);
        pointer-events: none;
    }
    
    .prediction-value {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 1rem 0;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        position: relative;
        font-family: 'Poppins', sans-serif;
        color: #ffffff;
    }
    
    .prediction-label {
        font-size: 0.9rem;
        opacity: 0.95;
        text-transform: uppercase;
        letter-spacing: 3px;
        font-weight: 600;
        position: relative;
        color: #f8fafc;
    }
    
    .prediction-box-alt {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    }
    
    /* Info Cards with Gradient Ribbons */
    .info-card {
        background: linear-gradient(to bottom right, #ffffff, #f8fafc);
        padding: 1.8rem;
        border-radius: 12px;
        border-left: 4px solid #1e3a8a;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        border-left-width: 6px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        transform: translateX(5px);
    }
    
    .info-card strong {
        color: #0f172a;
        font-size: 1.1rem;
        font-weight: 700;
    }
    
    .info-card small {
        color: #475569;
        line-height: 1.6;
    }
    
    /* Stat Cards with KPI Design */
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
        text-align: center;
        border-top: 3px solid #1e3a8a;
        transition: all 0.2s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
    }
    
    .stat-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0.5rem 0;
        font-family: 'Poppins', sans-serif;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    /* Tabs Styling with Glow Effect */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f1f5f9;
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border-radius: 10px;
        color: #475569;
        font-weight: 600;
        padding: 0 2rem;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(30, 58, 138, 0.1);
        color: #0f172a;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: #f8fafc;
        box-shadow: 0 4px 15px rgba(30, 58, 138, 0.3);
    }
    
    /* Select Box Styling with Focus Outline */
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        transition: all 0.3s ease;
        background-color: white;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #cbd5e1;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #1e3a8a;
        box-shadow: 0 0 0 3px rgba(30, 58, 138, 0.1);
    }
    
    .stSelectbox label {
        color: #0f172a;
        font-weight: 600;
    }
    
    /* Alert Boxes */
    .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
        animation: fadeIn 0.3s ease-in;
    }
    
    /* Sidebar Styling - Darker with Contrast */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #0d1b2a 0%, #1b263b 100%);
    }
    
    .css-1d391kg p, [data-testid="stSidebar"] p,
    .css-1d391kg h1, [data-testid="stSidebar"] h1,
    .css-1d391kg h2, [data-testid="stSidebar"] h2,
    .css-1d391kg h3, [data-testid="stSidebar"] h3 {
        color: #e2e8f0;
    }
    
    .css-1d391kg h2, [data-testid="stSidebar"] h2 {
        color: #f8fafc;
    }
    
    /* Sidebar Section Labels */
    [data-testid="stSidebar"] .element-container h3 {
        color: #94a3b8;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Sidebar Section Hover Effects */
    [data-testid="stSidebar"] .element-container {
        transition: all 0.2s ease;
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    [data-testid="stSidebar"] .element-container:hover {
        background-color: rgba(255, 255, 255, 0.05);
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.3);
    }
    
    /* Sidebar active/hover menu items */
    [data-testid="stSidebar"] .stRadio > label,
    [data-testid="stSidebar"] .stCheckbox > label {
        color: #e2e8f0;
    }
    
    [data-testid="stSidebar"] .stRadio > label:hover,
    [data-testid="stSidebar"] .stCheckbox > label:hover {
        color: #f1f5f9;
    }
    
    /* Loading Animation */
    .loading-text {
        text-align: center;
        font-size: 1.2rem;
        color: #1e3a8a;
        font-weight: 600;
        padding: 2rem;
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        padding: 0.35em 0.65em;
        font-size: 0.75rem;
        font-weight: 700;
        line-height: 1;
        color: #f8fafc;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 8px;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    }
    
    .badge-success {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: #ffffff;
    }
    
    .badge-danger {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: #ffffff;
    }
    
    .badge-warning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: #ffffff;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, #cbd5e1, transparent);
    }
    
    /* Status Indicators */
    .status-positive {
        color: #16a34a;
        font-weight: 600;
    }
    
    .status-negative {
        color: #94a3b8;
        font-weight: 600;
    }
    
    /* Metric labels and descriptions */
    .stMetric label {
        color: #475569 !important;
    }
    
    [data-testid="stSidebar"] .stMetric label {
        color: #94a3b8 !important;
    }
    
    [data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }
    
    /* Info boxes on sidebar */
    [data-testid="stSidebar"] .stAlert {
        background-color: rgba(30, 58, 138, 0.2);
        color: #e2e8f0;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    /* Caption text */
    .stCaption {
        color: #94a3b8;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        color: #0f172a;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .header-title {
            font-size: 2rem;
        }
        .prediction-value {
            font-size: 2.5rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Load models with error handling
@st.cache_resource
def load_models():
    """Load trained ML models with comprehensive error handling"""
    try:
        with open('models/best_blood_group_model.pkl', 'rb') as f:
            blood_model = pickle.load(f)
        with open('models/best_genotype_model.pkl', 'rb') as f:
            genotype_model = pickle.load(f)
        return blood_model, genotype_model, True, None
    except FileNotFoundError as e:
        return None, None, False, f"Model files not found: {str(e)}"
    except Exception as e:
        return None, None, False, f"Error loading models: {str(e)}"

blood_model, genotype_model, models_loaded, error_msg = load_models()

# Initialize label encoders for genotype features
@st.cache_resource
def get_label_encoders():
    """Initialize and fit label encoders for categorical features"""
    le_sickling = LabelEncoder()
    le_solubility = LabelEncoder()
    le_bands = LabelEncoder()
    
    le_sickling.fit(['No', 'Few', 'Yes'])
    le_solubility.fit(['Clear', 'Cloudy'])
    le_bands.fit(['A', 'A and C', 'A and S', 'C', 'S', 'S and C'])
    
    return le_sickling, le_solubility, le_bands

le_sickling, le_solubility, le_bands = get_label_encoders()

# Function to generate PDF report
def generate_pdf_report(patient_name, patient_gender, patient_age, anti_a, anti_b, anti_d,
                        sickling, solubility, bands, blood_group, genotype, 
                        bg_data, gt_data, current_time, report_date):
    """Generate a professional PDF report for blood group and genotype analysis"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1e3a8a'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#1e3a8a'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#475569'),
        spaceAfter=10,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=6,
        alignment=TA_LEFT
    )
    
    # Title
    title = Paragraph("BLOOD GROUP & GENOTYPE ANALYSIS REPORT", title_style)
    elements.append(title)
    elements.append(Spacer(1, 0.2*inch))
    
    # Report metadata
    meta_data = [
        [Paragraph("<b>Report Date:</b>", normal_style), Paragraph(report_date, normal_style)],
        [Paragraph("<b>Analysis Time:</b>", normal_style), Paragraph(current_time, normal_style)],
    ]
    meta_table = Table(meta_data, colWidths=[2*inch, 4*inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#0f172a')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1'))
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Patient Information
    elements.append(Paragraph("PATIENT INFORMATION", heading_style))
    patient_data = [
        [Paragraph("<b>Name:</b>", normal_style), Paragraph(patient_name if patient_name else 'Not Provided', normal_style)],
        [Paragraph("<b>Gender:</b>", normal_style), Paragraph(patient_gender, normal_style)],
        [Paragraph("<b>Age:</b>", normal_style), Paragraph(str(patient_age) if patient_age else 'Not Provided', normal_style)],
    ]
    patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#0f172a')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0'))
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Test Results
    elements.append(Paragraph("LABORATORY TEST RESULTS", heading_style))
    
    # Serological Tests
    elements.append(Paragraph("Serological Tests (Blood Group):", subheading_style))
    serology_data = [
        ['Test', 'Result'],
        ['Anti-A Reaction', 'Positive' if anti_a == 1 else 'Negative'],
        ['Anti-B Reaction', 'Positive' if anti_b == 1 else 'Negative'],
        ['Anti-D Reaction (Rh Factor)', 'Positive (Rh+)' if anti_d == 1 else 'Negative (Rh-)'],
    ]
    serology_table = Table(serology_data, colWidths=[3*inch, 3*inch])
    serology_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')])
    ]))
    elements.append(serology_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # Genotype Tests
    elements.append(Paragraph("Genotype Tests (Haemoglobin Analysis):", subheading_style))
    genotype_data = [
        ['Test', 'Result'],
        ['Sickling Test', sickling],
        ['Solubility Test', solubility],
        ['Electrophoresis Bands', bands],
    ]
    genotype_table = Table(genotype_data, colWidths=[3*inch, 3*inch])
    genotype_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')])
    ]))
    elements.append(genotype_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Analysis Results
    elements.append(Paragraph("ANALYSIS RESULTS", heading_style))
    
    results_data = [
        ['Blood Group', blood_group, 'ABO/Rh System'],
        ['Population Frequency', bg_data['frequency'], ''],
        ['', '', ''],
        ['Genotype', genotype, gt_data['severity']],
        ['Description', gt_data['description'], ''],
    ]
    results_table = Table(results_data, colWidths=[2*inch, 2*inch, 2*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, 0), colors.HexColor('#059669')),
        ('BACKGROUND', (1, 0), (-1, 0), colors.HexColor('#059669')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BACKGROUND', (0, 3), (0, 3), colors.HexColor('#ef4444')),
        ('BACKGROUND', (1, 3), (-1, 3), colors.HexColor('#ef4444')),
        ('TEXTCOLOR', (0, 3), (-1, 3), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1'))
    ]))
    elements.append(results_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Clinical Recommendations
    elements.append(Paragraph("CLINICAL RECOMMENDATIONS", heading_style))
    
    elements.append(Paragraph("Blood Transfusion Compatibility:", subheading_style))
    elements.append(Paragraph(bg_data['compatibility'], normal_style))
    elements.append(Paragraph("Always verify with cross-matching before transfusion. In emergency situations, use O- (universal donor).", normal_style))
    elements.append(Spacer(1, 0.2*inch))
    
    elements.append(Paragraph("Genotype Management:", subheading_style))
    if genotype in ['SS', 'SC']:
        recs = [
            "• Regular medical follow-up required",
            "• Avoid dehydration and extreme temperatures",
            "• Pain management protocol needed",
            "• Prophylactic antibiotics may be indicated"
        ]
    elif genotype in ['AS', 'AC']:
        recs = [
            "• Generally asymptomatic - carrier status",
            "• Genetic counseling recommended",
            "• Important for family planning considerations"
        ]
    else:
        recs = [
            "• Normal haemoglobin detected",
            "• No special management required",
            "• Continue standard health maintenance"
        ]
    
    for rec in recs:
        elements.append(Paragraph(rec, normal_style))
    
    elements.append(Spacer(1, 0.3*inch))
    
    # Footer
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#94a3b8'),
        alignment=TA_CENTER
    )
    elements.append(Paragraph("Clinical AI Platform v2.0 | Blood Group & Genotype Predictor", footer_style))
    elements.append(Paragraph("Powered by scikit-learn | Validated by clinical data", footer_style))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

# Header
st.markdown("""
    <div class='header-container'>
        <h1 class='header-title'>Blood Group & Genotype Predictor</h1>
        <p class='header-subtitle'>Enterprise-grade AI-powered clinical diagnostic platform for accurate blood analysis</p>
    </div>
""", unsafe_allow_html=True)

# Model Status Check
if not models_loaded:
    st.error(f"Model Error: {error_msg}")
    st.info("Please ensure model files are in the 'models/' directory or run the training script first.")
    st.code("python complete_blood_prediction_system.py", language="bash")
    st.stop()

# Patient Information Card
st.markdown("""
    <div class='card'>
        <div class='card-title'>Patient Information</div>
        <p style='color: #64748b; margin-bottom: 1.5rem;'>Enter patient details for the analysis report</p>
    </div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    patient_name = st.text_input(
        "Patient Name",
        placeholder="Enter full name",
        help="Patient's full legal name"
    )

with col2:
    patient_gender = st.selectbox(
        "Gender",
        options=["Male", "Female", "Other"],
        help="Patient's gender"
    )

with col3:
    patient_age = st.number_input(
        "Age",
        min_value=0,
        max_value=150,
        value=None,
        placeholder="Enter age",
        help="Patient's age in years"
    )

st.markdown("<br>", unsafe_allow_html=True)

# Input Form Card
st.markdown("""
    <div class='card'>
        <div class='card-title'>Blood Group Analysis</div>
        <p style='color: #64748b; margin-bottom: 1.5rem;'>Enter serological test results from antiserum reactions</p>
    </div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    anti_a = st.selectbox(
        "Anti-A Reaction",
        options=[0, 1],
        format_func=lambda x: "Positive (Agglutination)" if x == 1 else "Negative (No Reaction)",
        help="Agglutination with Anti-A serum indicates presence of A antigen"
    )
    if anti_a == 1:
        st.markdown('<p class="status-positive">Positive Result</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-negative">Negative Result</p>', unsafe_allow_html=True)

with col2:
    anti_b = st.selectbox(
        "Anti-B Reaction",
        options=[0, 1],
        format_func=lambda x: "Positive (Agglutination)" if x == 1 else "Negative (No Reaction)",
        help="Agglutination with Anti-B serum indicates presence of B antigen"
    )
    if anti_b == 1:
        st.markdown('<p class="status-positive">Positive Result</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-negative">Negative Result</p>', unsafe_allow_html=True)

with col3:
    anti_d = st.selectbox(
        "Anti-D Reaction (Rh Factor)",
        options=[0, 1],
        format_func=lambda x: "Positive (Rh+)" if x == 1 else "Negative (Rh-)",
        help="Rh factor test with Anti-D serum determines positive or negative blood type"
    )
    if anti_d == 1:
        st.markdown('<p class="status-positive">Positive Result</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-negative">Negative Result</p>', unsafe_allow_html=True)

# Sample validation
if anti_a == 0 and anti_b == 0 and anti_d == 0:
    st.warning("All reactions are negative. This sample may be invalid or diluted. Please verify test kit integrity.")

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
    <div class='card'>
        <div class='card-title'>Genotype Analysis</div>
        <p style='color: #64748b; margin-bottom: 1.5rem;'>Enter haemoglobin electrophoresis and variant analysis results</p>
    </div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    sickling = st.selectbox(
        "Sickling Test",
        options=['No', 'Few', 'Yes'],
        help="Presence of sickle-shaped red blood cells under reduced oxygen conditions"
    )

with col2:
    solubility = st.selectbox(
        "Solubility Test",
        options=['Clear', 'Cloudy'],
        help="HbS (sickle haemoglobin) is less soluble in phosphate buffer solutions"
    )

with col3:
    bands = st.selectbox(
        "Electrophoresis Bands",
        options=['A', 'A and S', 'S', 'A and C', 'C', 'S and C'],
        help="Haemoglobin variants visible on gel electrophoresis based on electrical charge"
    )

st.markdown("<br>", unsafe_allow_html=True)

# Predict button with validation
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("Analyze Results", type="primary", width='stretch')

if predict_btn:
    # Input validation
    if anti_a is None or anti_b is None or anti_d is None:
        st.error("Please complete all blood group test fields")
    elif sickling is None or solubility is None or bands is None:
        st.error("Please complete all genotype test fields")
    else:
        # Show loading animation
        with st.spinner('Analyzing laboratory results...'):
            time.sleep(1.5)  # Simulate processing time
            
            try:
                # Predict blood group
                X_blood = [[anti_a, anti_b, anti_d]]
                if blood_model is not None:
                    blood_group = blood_model.predict(X_blood)[0]
                else:
                    st.error("Blood group model not available")
                    st.stop()
                
                # Predict genotype
                sickling_enc = le_sickling.transform([sickling])[0]
                solubility_enc = le_solubility.transform([solubility])[0]
                bands_enc = le_bands.transform([bands])[0]
                X_genotype = [[sickling_enc, solubility_enc, bands_enc]]
                if genotype_model is not None:
                    genotype = genotype_model.predict(X_genotype)[0]
                else:
                    st.error("Genotype model not available")
                    st.stop()
                
                # Success message
                st.success("Analysis complete! Results are ready.")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                        <div class='prediction-box'>
                            <div class='prediction-label'>Blood Group</div>
                            <div class='prediction-value'>{blood_group}</div>
                            <div style='margin-top: 1rem; opacity: 0.9;'>
                                <span class='badge badge-success'>ABO/Rh System</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                        <div class='prediction-box prediction-box-alt'>
                            <div class='prediction-label'>Genotype</div>
                            <div class='prediction-value'>{genotype}</div>
                            <div style='margin-top: 1rem; opacity: 0.9;'>
                                <span class='badge badge-danger'>Haemoglobin Variant</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Clinical interpretation
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""
                    <div class='card'>
                        <div class='card-title'>Clinical Interpretation</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Blood group info
                blood_info = {
                    'A+': {'compatibility': 'Can receive: A+, A-, O+, O- | Can donate to: A+, AB+', 'frequency': '35.7%'},
                    'A-': {'compatibility': 'Can receive: A-, O- | Can donate to: A+, A-, AB+, AB-', 'frequency': '6.3%'},
                    'B+': {'compatibility': 'Can receive: B+, B-, O+, O- | Can donate to: B+, AB+', 'frequency': '8.5%'},
                    'B-': {'compatibility': 'Can receive: B-, O- | Can donate to: B+, B-, AB+, AB-', 'frequency': '1.5%'},
                    'AB+': {'compatibility': 'Universal Recipient (can receive from all) | Can donate to: AB+', 'frequency': '3.4%'},
                    'AB-': {'compatibility': 'Can receive: AB-, A-, B-, O- | Can donate to: AB+, AB-', 'frequency': '0.6%'},
                    'O+': {'compatibility': 'Can receive: O+, O- | Can donate to: A+, B+, AB+, O+', 'frequency': '37.4%'},
                    'O-': {'compatibility': 'Universal Donor (can donate to all) | Can receive: O- only', 'frequency': '6.6%'}
                }
                
                genotype_info = {
                    'AA': {'description': 'Normal haemoglobin - No sickle cell trait', 'severity': 'Normal', 'color': '#28a745'},
                    'AS': {'description': 'Sickle cell trait - Carrier, usually asymptomatic', 'severity': 'Trait', 'color': '#ffc107'},
                    'SS': {'description': 'Sickle cell disease - Requires medical management', 'severity': 'Disease', 'color': '#dc3545'},
                    'AC': {'description': 'Haemoglobin C trait - Usually asymptomatic', 'severity': 'Trait', 'color': '#17a2b8'},
                    'SC': {'description': 'HbSC disease - Mild to moderate sickle cell disease', 'severity': 'Disease', 'color': '#fd7e14'},
                    'CC': {'description': 'Haemoglobin C disease - Mild anaemia possible', 'severity': 'Disease', 'color': '#6610f2'}
                }
                
                col1, col2 = st.columns(2)
                
                with col1:
                    bg_data = blood_info.get(blood_group, {'compatibility': 'Information not available', 'frequency': 'N/A'})
                    st.markdown(f"""
                        <div class='info-card'>
                            <strong>Blood Group {blood_group}</strong><br>
                            <small>Compatibility: {bg_data['compatibility']}</small><br>
                            <small>Population Frequency: <strong>{bg_data['frequency']}</strong></small>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    gt_data = genotype_info.get(genotype, {'description': 'Information not available', 'severity': 'Unknown', 'color': '#6c757d'})
                    st.markdown(f"""
                        <div class='info-card' style='border-left-color: {gt_data['color']};'>
                            <strong>Genotype {genotype}</strong><br>
                            <small>{gt_data['description']}</small><br>
                            <small>Status: <span style='color: {gt_data['color']}; font-weight: 700;'>{gt_data['severity']}</span></small>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Detailed clinical recommendations
                st.markdown("<br>", unsafe_allow_html=True)
                
                with st.expander("View Detailed Clinical Recommendations", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Blood Transfusion Guidelines")
                        st.markdown(f"""
                        **For {blood_group} patients:**
                        - {bg_data['compatibility']}
                        - Always verify with cross-matching
                        - Emergency: Use O- (universal donor)
                        """)
                    
                    with col2:
                        st.markdown("### Genotype Management")
                        if genotype in ['SS', 'SC']:
                            st.warning("""
                            **Sickle Cell Disease Detected**
                            - Regular medical follow-up required
                            - Avoid dehydration and extreme temperatures
                            - Pain management protocol needed
                            - Prophylactic antibiotics may be indicated
                            """)
                        elif genotype in ['AS', 'AC']:
                            st.info("""
                            **Carrier Status**
                            - Generally asymptomatic
                            - Genetic counseling recommended
                            - Important for family planning
                            """)
                        else:
                            st.success("""
                            **Normal Haemoglobin**
                            - No special management required
                            - Standard health maintenance
                            """)
        
                # Timestamp and Download Report
                st.markdown("<br>", unsafe_allow_html=True)
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                report_date = datetime.now().strftime("%Y-%m-%d")
                st.caption(f"Analysis completed at: {current_time}")
                
                # Generate PDF report
                st.markdown("<br>", unsafe_allow_html=True)
                
                pdf_buffer = generate_pdf_report(
                    patient_name=patient_name if patient_name else '',
                    patient_gender=patient_gender,
                    patient_age=patient_age,
                    anti_a=anti_a,
                    anti_b=anti_b,
                    anti_d=anti_d,
                    sickling=sickling,
                    solubility=solubility,
                    bands=bands,
                    blood_group=blood_group,
                    genotype=genotype,
                    bg_data=bg_data,
                    gt_data=gt_data,
                    current_time=current_time,
                    report_date=report_date
                )
                
                # Download button
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    filename = f"blood_analysis_report_{patient_name.replace(' ', '_') if patient_name else 'patient'}_{report_date}.pdf"
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_buffer,
                        file_name=filename,
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True
                    )
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.info("Please check your input values and try again.")

