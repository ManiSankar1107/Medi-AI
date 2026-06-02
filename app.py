import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
#fpdf
from fpdf import FPDF
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import PIL
import streamlit as st
import numpy as np
import joblib
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def is_brain_mri(pic_file):
    import numpy as np
    import cv2
    from PIL import Image
    try:
        img = Image.open(pic_file).convert('RGB')
        pic_file.seek(0)
    except:
        return False
    
    img_array = np.array(img)
    if img_array.size == 0:
        return False
        
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    r_int, g_int, b_int = r.astype(np.int32), g.astype(np.int32), b.astype(np.int32)
    diff_rg = np.abs(r_int - g_int)
    diff_gb = np.abs(g_int - b_int)
    diff_br = np.abs(b_int - r_int)
    mean_diff = (np.mean(diff_rg) + np.mean(diff_gb) + np.mean(diff_br)) / 3.0
    
    # 1. Check if Grayscale (Brain MRIs and Lung CTs are grayscale)
    if mean_diff > 15:
        return False
        
    # 2. Check Structural Shape via OpenCV
    # Brain MRIs are generally a single solid mass (high compactness)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return False
        
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    
    if perimeter == 0:
        return False
        
    compactness = (4 * np.pi * area) / (perimeter * perimeter)
    
    # Brain outlines are very solid ovals > 0.60
    if compactness < 0.50:
        return False
        
    return True

def is_lung_scan(pic_file):
    import numpy as np
    import cv2
    from PIL import Image
    try:
        img = Image.open(pic_file).convert('RGB')
        pic_file.seek(0)
    except:
        return False
        
    img_array = np.array(img)
    if img_array.size == 0:
        return False
        
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    r_int, g_int, b_int = r.astype(np.int32), g.astype(np.int32), b.astype(np.int32)
    diff_rg = np.abs(r_int - g_int)
    diff_gb = np.abs(g_int - b_int)
    diff_br = np.abs(b_int - r_int)
    mean_diff = (np.mean(diff_rg) + np.mean(diff_gb) + np.mean(diff_br)) / 3.0
    
    # 1. Check if Grayscale
    if mean_diff > 15:
        return False
        
    return True

def is_skin_image(pic_file):
    import numpy as np
    from PIL import Image
    try:
        img = Image.open(pic_file).convert('RGB')
        pic_file.seek(0)
    except:
        return False
        
    img_array = np.array(img)
    if img_array.size == 0:
        return False
        
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    r_int, g_int, b_int = r.astype(np.int32), g.astype(np.int32), b.astype(np.int32)
    diff_rg = np.abs(r_int - g_int)
    diff_gb = np.abs(g_int - b_int)
    diff_br = np.abs(b_int - r_int)
    mean_diff = (np.mean(diff_rg) + np.mean(diff_gb) + np.mean(diff_br)) / 3.0
    
    # 1. Grayscale images (diff <= 10) like MRIs or Lung X-Rays should fail
    if mean_diff <= 10:
        return False
        
    # 2. Check Color Ratios for Skin vs Blood
    # Skin has high red/brown, Blood smears have high purple/blue/pink stains
    r_sum = np.sum(r_int, dtype=np.float64) + 1
    g_sum = np.sum(g_int, dtype=np.float64) + 1
    b_sum = np.sum(b_int, dtype=np.float64) + 1
    total = r_sum + g_sum + b_sum
    
    # Skin naturally has a dominant red channel compared to blue
    # Blood slides usually have strong blues/purples
    if (b_sum / total) > 0.32:  # Too much blue = likely blood stain
        return False
        
    return True

def is_blood_image(pic_file):
    import numpy as np
    from PIL import Image
    try:
        img = Image.open(pic_file).convert('RGB')
        pic_file.seek(0)
    except:
        return False
        
    img_array = np.array(img)
    if img_array.size == 0:
        return False
        
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    r_int, g_int, b_int = r.astype(np.int32), g.astype(np.int32), b.astype(np.int32)
    diff_rg = np.abs(r_int - g_int)
    diff_gb = np.abs(g_int - b_int)
    diff_br = np.abs(b_int - r_int)
    mean_diff = (np.mean(diff_rg) + np.mean(diff_gb) + np.mean(diff_br)) / 3.0
    
    # 1. Grayscale images should fail
    if mean_diff <= 10:
        return False
        
    # 2. Check Color Ratios for Blood Stains
    r_sum = np.sum(r_int, dtype=np.float64) + 1
    g_sum = np.sum(g_int, dtype=np.float64) + 1
    b_sum = np.sum(b_int, dtype=np.float64) + 1
    total = r_sum + g_sum + b_sum
    
    # Blood images usually have noticeable blue/purple stains 
    # Or strong pinkish hues across the board with white/bluish background gaps.
    if (b_sum / total) < 0.28 and (r_sum / total) > 0.40:
        # Too much red with very little blue = likely skin lesion
        return False
    
    return True

def box(disease_name):
    return f"""
        <div style="
            background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-size: 24px;
            font-weight: 700;
            color: #ffffff;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 10px;">
            {disease_name}
        </div>
    """
def symbox(disease_name):
    return f"""
        <div style="
            background-color: #fdf2f8;
            border-left: 5px solid #ec4899;
            padding: 15px;
            border-radius: 8px;
            font-size: 16px;
            color: #1f2937;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            line-height: 1.6;
            font-weight: 500;">
            {disease_name}
        </div>
    """
def descbox(disease_name):
    return f"""
        <div style="
            background-color: #fefce8;
            border-left: 5px solid #eab308;
            padding: 15px;
            border-radius: 8px;
            font-size: 16px;
            color: #1f2937;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            text-align: justify;
            line-height: 1.6;
            font-weight: 500;">
            {disease_name}
        </div>
    """
def prevbox(disease_name):
    return f"""
        <div style="
            background-color: #f0fdfa;
            border-left: 5px solid #14b8a6;
            padding: 15px;
            border-radius: 8px;
            font-size: 16px;
            color: #1f2937;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            text-align: justify;
            line-height: 1.6;
            font-weight: 500;">
            {disease_name}
        </div>
    """
def load_model():
    brain_model = tf.keras.models.load_model("brain_model.h5", compile=False)
    brain_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return brain_model

st.markdown("""
<style>
/* Custom Animated Hamburger Menu for Sidebar */
[data-testid="stSidebarCollapseButton"] svg {
    display: none !important;
}
[data-testid="stSidebarCollapseButton"] {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 2rem;
    height: 2rem;
}
[data-testid="stSidebarCollapseButton"]::before {
    content: '';
    position: absolute;
    width: 18px;
    height: 2px;
    border-radius: 2px;
    background-color: #555;
    box-shadow: 0 -6px 0 0 #555, 0 6px 0 0 #555;
    transition: all 0.3s cubic-bezier(0.25, 0.1, 0.25, 1);
}
[data-testid="stSidebarCollapseButton"]:hover::before {
    background-color: #0f62fe;
    box-shadow: 0 -6px 0 0 #0f62fe, 0 6px 0 0 #0f62fe;
    transform: scale(1.1);
}

[data-testid="collapsedControl"] svg {
    display: none !important;
}
[data-testid="collapsedControl"] {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 2rem;
    height: 2rem;
}
[data-testid="collapsedControl"]::before {
    content: '';
    position: absolute;
    width: 18px;
    height: 2px;
    border-radius: 2px;
    background-color: #555;
    box-shadow: 0 -6px 0 0 #555, 0 6px 0 0 #555;
    transition: all 0.3s cubic-bezier(0.25, 0.1, 0.25, 1);
}
[data-testid="collapsedControl"]:hover::before {
    background-color: #0f62fe;
    box-shadow: 0 -6px 0 0 #0f62fe, 0 6px 0 0 #0f62fe;
    transform: scale(1.1);
}

/* File Uploader Custom Styling */
[data-testid="stFileUploader"] {
    background-color: rgba(255, 255, 255, 0.9) !important;
    border: 2px solid white !important;
    border-radius: 12px !important;
    padding: 10px !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
    transition: all 0.3s ease-in-out !important;
}
[data-testid="stFileUploader"]:hover {
    background-color: #ffffff !important;
    border-color: white !important;
    box-shadow: 0 8px 25px rgba(255, 255, 255, 0.5) !important;
    transform: translateY(-2px);
}
[data-testid="stMarkdownContainer"] p {
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <img src="https://cdn-icons-png.flaticon.com/512/2966/2966327.png" width="80" style="margin-bottom: 10px;">
            <h2 style="font-weight: 800; color: #10bec4; margin: 0; font-family: 'Inter', sans-serif, system-ui;">Medi<span style="color: #555;">Ai</span></h2>
            <p style="color: gray; font-size: 0.85rem; margin-top: 5px; font-weight: 500;">Medical Diagnosis Portal</p>
            <hr style="margin-top: 20px; margin-bottom: 0px; border: 0; border-top: 1px solid rgba(128,128,128,0.2);">
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📋 Patient Details")
    if 'patient_name' not in st.session_state:
        st.session_state['patient_name'] = ""
    if 'patient_age' not in st.session_state:
        st.session_state['patient_age'] = 25
    if 'patient_gender' not in st.session_state:
        st.session_state['patient_gender'] = "Male"

    st.session_state['patient_name'] = st.text_input("Patient Name", st.session_state['patient_name'])
    pcol1, pcol2 = st.columns(2)
    st.session_state['patient_age'] = pcol1.number_input("Age", min_value=1, max_value=120, value=st.session_state['patient_age'])
    st.session_state['patient_gender'] = pcol2.selectbox("Gender", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(st.session_state['patient_gender']))
    
    st.markdown("<hr style='margin-top: 10px; margin-bottom: 10px; border: 0; border-top: 1px solid rgba(128,128,128,0.2);'>", unsafe_allow_html=True)
    
    select = option_menu(
        "",
        ['Home',"Brain Disease",'Skin Disease', 'Lung Disease',"Heart Disease",'Blood Disease','Liver Disease','Report'],
        icons=['house-fill','reddit','people-fill','lungs-fill','heart-pulse-fill','droplet-fill','activity','chat-left-quote-fill'],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding": "8px", "background-color": "#d6d6d6", "border-radius": "10px"}, 
            "icon": {"color": "black", "font-size": "20px"},    
            "nav-link": {
                "font-size": "16px",
                "margin": "6px 0px",
                "color": "black",
                "border-radius": "8px",
                "--hover-color": "#f1f5f9",
                "transition": "all 0.3s ease-in-out"
            },   
            "nav-link-selected": {
                "background-color": "#10bec4",
                "color": "white",
                "box-shadow": "0 4px 10px rgba(16, 190, 196, 0.4)"
            },
        },
    )
if select=='Home':
    st.markdown(
        """
        <style>
        /* Override default Streamlit top padding */
        div.block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 2rem !important;
        }
        .hero-container {
            padding: 0.5rem 0 1.5rem 0;
            text-align: center;
        }
        .hero-title {
            font-size: 3rem;
            font-weight: 800;
            color: #10bec4; /* matches the sidebar selected color */
            text-align: center;
            margin-bottom: 15px;
            letter-spacing: -0.5px;
        }
        .hero-subtitle {
            text-align: center;
            font-size: 1.2rem;
            color: var(--text-color);
            opacity: 0.8;
            margin-bottom: 40px;
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
            line-height: 1.6;
        }
        .feature-box {
            background-color: var(--secondary-background-color);
            padding: 30px 20px;
            border-radius: 20px;
            text-align: center;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            height: 100%;
            border: 1px solid rgba(16, 190, 196, 0.1);
            position: relative;
            overflow: hidden;
            z-index: 1;
        }
        .feature-box:hover {
            transform: translateY(-5px) scale(1.02);
            border-color: rgba(16, 190, 196, 0.3);
            box-shadow: 0 12px 25px rgba(0, 0, 0, 0.06), 0 0 20px rgba(16, 190, 196, 0.1);
        }
        .icon-wrapper {
            width: 75px;
            height: 75px;
            background: rgba(16, 190, 196, 0.1);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 20px auto;
            font-size: 2.2rem;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }
        .feature-box:hover .icon-wrapper {
            background-color: #10bec4;
            transform: scale(1.1) rotate(5deg);
            box-shadow: 0 8px 15px rgba(16, 190, 196, 0.25);
        }
        .feature-title {
            font-size: 1.2rem;
            font-weight: 700;
            color: var(--text-color);
            margin-bottom: 10px;
        }
        .feature-desc {
            font-size: 0.95rem;
            color: var(--text-color);
            opacity: 0.7;
            line-height: 1.5;
        }
        </style>
        
        <div class="hero-container">
            <div class="hero-title">AI Engine for Healthcare Diagnostics</div>
            <div class="hero-subtitle">Empowering medical professionals with state-of-the-art artificial intelligence for rapid, accurate, and reliable disease prediction across multiple domains.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🔍 Diagnostic Modules Overview")
    st.write("") # spacer
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <div class="icon-wrapper">🧠</div>
            <div class="feature-title">Brain Disease</div>
            <div class="feature-desc">Analyze MRI scans to detect and classify Alzheimer's and evaluate dementia stages.</div>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        st.markdown("""
        <div class="feature-box">
            <div class="icon-wrapper">🫀</div>
            <div class="feature-title">Heart Disease</div>
            <div class="feature-desc">Evaluate comprehensive clinical metrics to carefully assess cardiovascular health and risks.</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="feature-box">
            <div class="icon-wrapper">🔬</div>
            <div class="feature-title">Skin Disease</div>
            <div class="feature-desc">Classify dermoscopic images into various benign and malignant skin lesion categories.</div>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        st.markdown("""
        <div class="feature-box">
            <div class="icon-wrapper">🩸</div>
            <div class="feature-title">Blood Disease</div>
            <div class="feature-desc">Examine microscopic blood smears to identify leukemia and other blood anomalies.</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-box">
            <div class="icon-wrapper">🫁</div>
            <div class="feature-title">Lung Disease</div>
            <div class="feature-desc">Process CT and X-Ray scans to detect pulmonary carcinomas and other respiratory issues.</div>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        st.markdown("""
        <div class="feature-box">
            <div class="icon-wrapper">🩺</div>
            <div class="feature-title">Liver Disease</div>
            <div class="feature-desc">Predict hepatic conditions by analyzing user patient background and biochemistry test results.</div>
        </div>
        """, unsafe_allow_html=True)

elif select=='Brain Disease':
    st.markdown("<h1 style='text-align: center;color:maroon;'>Brain Diseases Detection</h1>", unsafe_allow_html=True)
    # Background removed for professional UI
    pic = st.file_uploader(
    label="Upload an image",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=False,
    help="Upload an image of a brain scan to predict the stage of Alzheimer's Disease",
    )
    col1,col2,col3=st.columns([2,2,1])
    if col2.button("Predict",type='primary'):
        if pic is None:
            st.error("Please upload an image file")
        elif not is_brain_mri(pic):
            st.error("Invalid input. Please upload a brain MRI scan.")
        else:
            cols = st.columns([1, 2])
            with cols[1]:
                brain_model = load_model()
                labels = [
                    "Mild Dementia",
                    "Moderate Dementia",
                    "No Dementia",
                    "Very Mild Dementia",
                ]
                with st.spinner("Predicting..."):
                    img = PIL.Image.open(pic)
                    img = img.convert("RGB")
                    img = img.resize((128, 128))
                    img = tf.expand_dims(img, axis=0)

                    prediction = brain_model.predict(img)
                    prediction = tf.nn.softmax(prediction)

                    score = tf.reduce_max(prediction)
                    score = tf.round(score * 100)

                    prediction = tf.argmax(prediction, axis=1)
                    prediction: int = prediction.numpy()[0]

                    result = labels[prediction]

            #open report.csv file
            rp = pd.read_csv('report.csv')
            rp.loc[rp['DiseaseType'] == 'brain', 'Class'] = result
            rp.to_csv('report.csv', index=False)
            data = pd.read_csv('info.csv',encoding='utf-8')
            df = data[data['Name'] == result]
            col1,col2=st.columns([1,1])
            col1.image(pic, caption=pic.name, use_column_width=True)
            col2.markdown(box(result), unsafe_allow_html=True)
            col2.markdown("<h3 style='color: #4b5563; margin-top: 15px; margin-bottom: 5px; font-weight: 600;'>🩺 Symptoms</h3>", unsafe_allow_html=True)
            col2.markdown(symbox(df['Sym'].values[0]), unsafe_allow_html=True)
            # Display description
            col1,col2=st.columns([1,1])
            col1.markdown("<h3 style='color: #4b5563; margin-top: 15px; margin-bottom: 5px; font-weight: 600;'>📋 Description</h3>", unsafe_allow_html=True)
            col1.markdown(descbox(df['Desc'].values[0]), unsafe_allow_html=True)
            # Display symptoms           
            # Display prevention measures
            col2.markdown("<h3 style='color: #4b5563; margin-top: 15px; margin-bottom: 5px; font-weight: 600;'>🛡️ Prevention Measures</h3>", unsafe_allow_html=True)
            col2.markdown(prevbox(df['Prev'].values[0]), unsafe_allow_html=True)

elif select=='Skin Disease':
    # Background removed for professional UI
    st.markdown("<h1 style='text-align: center;color:blue;'>Skin Diseases Detection</h1>", unsafe_allow_html=True)
    skin_model=tf.keras.models.load_model("skin_model.h5", compile=False)
    skin_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    pic = st.file_uploader(
        label="Upload a Image",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
        help="Upload a picture of your skin to get a diagnosis",
    )
    col1,col2,col3=st.columns([1.5,1,1])
    if col2.button("Predict",type='primary'):
        if not pic:
            st.error("Please upload an image")
        elif not is_skin_image(pic):
            st.error("Invalid input. Please upload a valid skin lesion image.")
        else:
            cols = st.columns([2, 2])

            with cols[1]:
                labels = [
                    "actinic keratosis",
                    "basal cell carcinoma",
                    "dermatofibroma",
                    "melanoma",
                    "nevus",
                    "pigmented benign keratosis",
                    "seborrheic keratosis",
                    "squamous cell carcinoma (skin)",
                    "vascular lesion",
                ]

                with st.spinner("Processing image..."):
                    img = PIL.Image.open(pic)
                    img = img.resize((180, 180))
                    img = tf.keras.preprocessing.image.img_to_array(img)
                    img = tf.expand_dims(img, axis=0)

                    prediction = skin_model.predict(img)
                    prediction = tf.nn.softmax(prediction)

                    score = tf.reduce_max(prediction)
                    score = tf.round(score * 100, 2)

                with st.spinner("Predicting..."):
                    prediction = tf.argmax(prediction, axis=1)
                    prediction = prediction.numpy()
                    prediction = prediction[0]
                    #Precancerous Lesions: Actinic keratosis, Benign Lesions: Nevus, seborrheic keratosis, pigmented benign keratosis, dermatofibroma, Other Lesions: Vascular lesions cancer: basal cell carcinoma, squamous cell carcinoma, melanoma 
                    disease = str(labels[prediction]).title()
                    #add stage of cancer like cancer, precancerous, benign, other lesions cancer
                    if disease in ["Basal Cell Carcinoma", "Squamous Cell Carcinoma", "Melanoma"]:
                        type = "Cancereous disease"
                    elif disease in ["Nevus", "Seborrheic Keratosis", "Pigmented Benign Keratosis", "Dermatofibroma"]:
                        type = "Benign Lesions Cancer"
                    elif disease in ["Vascular Lesion"]:
                        type = "Other Lesions Cancer"
                    elif disease in ["Actinic Keratosis"]:
                        type = "Precancerous Lesion Cancer"
            #open report.csv file
            rp = pd.read_csv('report.csv')
            rp.loc[rp['DiseaseType'] == 'skin', 'Class'] = disease
            rp.to_csv('report.csv', index=False)
            # give a brief description of the disease
            data = pd.read_csv('info.csv', encoding='utf-8')
            df = data[data['Name'] == disease]
            col1,col2=st.columns([1,1])
            col1.image(pic, caption=type, use_column_width=True)
            col2.markdown(box(disease), unsafe_allow_html=True)
            col2.markdown("<h3 style='color: #4b5563; margin-top: 15px; margin-bottom: 5px; font-weight: 600;'>🩺 Symptoms</h3>", unsafe_allow_html=True)
            col2.markdown(symbox(df['Sym'].values[0]), unsafe_allow_html=True)
            # Display description
            col1,col2=st.columns([1,1])
            col1.markdown("<h3 style='color: #4b5563; margin-top: 15px; margin-bottom: 5px; font-weight: 600;'>📋 Description</h3>", unsafe_allow_html=True)
            col1.markdown(descbox(df['Desc'].values[0]), unsafe_allow_html=True)
            # Display symptoms           
            # Display prevention measures
            col2.markdown("<h3 style='color: #4b5563; margin-top: 15px; margin-bottom: 5px; font-weight: 600;'>🛡️ Prevention Measures</h3>", unsafe_allow_html=True)
            col2.markdown(prevbox(df['Prev'].values[0]), unsafe_allow_html=True)
elif select=='Lung Disease':
    # Background removed for professional UI
    st.markdown("<h1 style='text-align: center;color:darkgreen;'>Lung Disease Detection</h1>", unsafe_allow_html=True)
    model_path = "ct_effnet_best_model.hdf5"  # Update with your model path
    from tensorflow.keras.models import load_model
    lung_model = load_model(model_path, compile=False)
    lung_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Define class labels
    class_labels = {
        0: 'Adenocarcinoma',
        1: 'Large Cell Carcinoma',
        2: 'Normal Lung',
        3: 'Squamous Cell Carcinoma (Lung)'
    }

    # Function to preprocess the uploaded image
    def preprocess_image(image_file):
        img = Image.open(image_file)
        img = img.convert("RGB")  # Convert to RGB format
        img = img.resize((350, 350))  # Adjust the target size to match your model's input size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize pixel values to the range [0, 1]
        return img_array

    # Function to make predictions
    def predict(image):
        predictions = lung_model.predict(image)
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_labels.get(predicted_class_index, "Unknown")
        return predicted_class
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        if not is_lung_scan(uploaded_file):
            st.error("Invalid input. Please upload a valid lung CT/X-Ray scan.")
        else:
            image_array = preprocess_image(uploaded_file)
            prediction = predict(image_array)
            # give a brief description of the disease
            #open report.csv file
            rp = pd.read_csv('report.csv')
            rp.loc[rp['DiseaseType'] == 'lung', 'Class'] = prediction
            rp.to_csv('report.csv', index=False)
            data = pd.read_csv('info.csv', encoding='utf-8')
            df = data[data['Name'] == prediction]
            col1,col2=st.columns([1,1])
            col1.image(uploaded_file, caption='lung_image', use_column_width=True)
            col2.markdown(box(prediction), unsafe_allow_html=True)
            col2.markdown("<h3 style='color: #4b5563; margin-top: 15px; margin-bottom: 5px; font-weight: 600;'>🩺 Symptoms</h3>", unsafe_allow_html=True)
            col2.markdown(symbox(df['Sym'].values[0]), unsafe_allow_html=True)
            # Display description
            col1,col2=st.columns([1,1])
            col1.markdown("<h3 style='color: #4b5563; margin-top: 15px; margin-bottom: 5px; font-weight: 600;'>📋 Description</h3>", unsafe_allow_html=True)
            col1.markdown(descbox(df['Desc'].values[0]), unsafe_allow_html=True)
            # Display symptoms           
            # Display prevention measures
            col2.markdown("<h3 style='color: #4b5563; margin-top: 15px; margin-bottom: 5px; font-weight: 600;'>🛡️ Prevention Measures</h3>", unsafe_allow_html=True)
            col2.markdown(prevbox(df['Prev'].values[0]), unsafe_allow_html=True)

elif select=='Heart Disease':
    st.markdown("<h1 style='text-align: center;color: #89ad07;'>Heart Disease Detection</h1>", unsafe_allow_html=True)
    heart_model = joblib.load(open('model_pkl', 'rb'))
    # Background removed for professional UI
    # Input form
    st.info(f"Using Patient Details from sidebar: Age: {st.session_state.get('patient_age', 25)}, Gender: {st.session_state.get('patient_gender', 'Male')}")
    age = st.session_state.get('patient_age', 25)
    sex = st.session_state.get('patient_gender', 'Male')
    if sex not in ["Female", "Male"]:
        sex = "Male" # Clinical fallback
    
    with st.form("prediction_form"):
        col1,col2=st.columns(2)
        cp = col1.selectbox("Chest Pain Type:", options=["No Pain", "Low Pain", "High Pain", "Severe Pain"], index=0)
        trestbps = col2.number_input("Resting Blood Pressure (mm Hg):", min_value=80, max_value=200, value=120)
        
        col1,col2=st.columns(2)
        chol = col1.number_input("Cholesterol (mg/dL):", min_value=100, max_value=500, value=200)
        fbs = col2.selectbox("Fasting Blood Sugar > 120 mg/dL:", options=["No", "Yes"], index=0)
        
        col1,col2=st.columns(2)
        restecg = col1.selectbox("Rest ECG Results:", options=["0", "1", "2"], index=0)
        thalach = col2.number_input("Max Heart Rate Achieved:", min_value=50, max_value=220, value=150)
        
        col1,col2=st.columns(2)
        exang = col1.selectbox("Exercise-Induced Angina:", options=["No", "Yes"], index=0)
        oldpeak = col2.number_input("ST Depression Induced by Exercise:", min_value=0.0, max_value=10.0, step=0.1, value=0.0)
        
        col1,col2=st.columns(2)
        slope = col1.selectbox("Slope of Peak Exercise ST Segment:", options=["0", "1", "2"], index=0)
        ca = col2.selectbox("Number of Major Vessels (0-4):", options=["0", "1", "2", "3", "4"], index=0)
        
        thal = st.selectbox("Thalassemia (0-3):", options=["0", "1", "2", "3"], index=0)

        # Submit button
        submitted = st.form_submit_button("Predict")

    if submitted:
        # Map string inputs to numerical values
        cp_mapping = {"No Pain": 0, "Low Pain": 1, "High Pain": 2, "Severe Pain": 3}
        sex_mapping = {"Female": 0, "Male": 1}
        fbs_mapping = {"No": 0, "Yes": 1}
        exang_mapping = {"No": 0, "Yes": 1}

        # Convert inputs into the expected format
        input_features = [
            age,
            sex_mapping[sex],
            cp_mapping[cp],
            trestbps,
            chol,
            fbs_mapping[fbs],
            int(restecg),
            thalach,
            exang_mapping[exang],
            oldpeak,
            int(slope),
            int(ca),
            int(thal)
        ]
        features_value = [np.array(input_features)]
        features_name = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang",
                        "oldpeak", "slope", "ca", "thal"]

        # Create DataFrame
        df = pd.DataFrame(features_value, columns=features_name)

        # Make prediction
        prediction = heart_model.predict(df)
        col1,col2,col3=st.columns([3,6,3])
        # Display the result
        clas=['Normal Heart','Heart Disease']
        probabilities = prediction[0]
        prediction=clas[probabilities]
        #open report.csv file
        rp = pd.read_csv('report.csv')
        rp.loc[rp['DiseaseType'] == 'heart', 'Class'] = prediction
        rp.to_csv('report.csv', index=False)
        data = pd.read_csv('info.csv', encoding='utf-8')
        df = data[data['Name'] == prediction]
        if prediction == 'Normal Heart':
            col1,col2=st.columns([1,1])
            pic='https://media.istockphoto.com/id/1128931450/photo/heart-attack-concept.jpg?s=612x612&w=0&k=20&c=XHOhTXhpZMSV6XIhXLbH6uvNQjZQS93b1UetGfqQXtI='
            col1.image(pic, caption=type, use_column_width=True)
            col2.markdown(box('Normal Heart'), unsafe_allow_html=True)
            col2.markdown("<h3 style='color: #4b5563; margin-top: 15px; margin-bottom: 5px; font-weight: 600;'>🩺 Symptoms</h3>", unsafe_allow_html=True)
            col2.markdown(symbox(df['Sym'].values[0]), unsafe_allow_html=True)
            # Display description
            col1,col2=st.columns([1,1])
            col1.markdown("<h3 style='color: #4b5563; margin-top: 15px; margin-bottom: 5px; font-weight: 600;'>📋 Description</h3>", unsafe_allow_html=True)
            col1.markdown(descbox(df['Desc'].values[0]), unsafe_allow_html=True)
            # Display symptoms           
            # Display prevention measures
            col2.markdown("<h3 style='color: #4b5563; margin-top: 15px; margin-bottom: 5px; font-weight: 600;'>🛡️ Prevention Measures</h3>", unsafe_allow_html=True)
            col2.markdown(prevbox(df['Prev'].values[0]), unsafe_allow_html=True)
        else:
            col1,col2=st.columns([1,1])
            pic='https://media.istockphoto.com/id/1128931450/photo/heart-attack-concept.jpg?s=612x612&w=0&k=20&c=XHOhTXhpZMSV6XIhXLbH6uvNQjZQS93b1UetGfqQXtI='
            col1.image(pic, caption=type, use_column_width=True)
            col2.markdown(box(prediction.upper()), unsafe_allow_html=True)
            col2.markdown("<h3 style='color: #4b5563; margin-top: 15px; margin-bottom: 5px; font-weight: 600;'>🩺 Symptoms</h3>", unsafe_allow_html=True)
            col2.markdown(symbox(df['Sym'].values[0]), unsafe_allow_html=True)
            # Display description
            col1,col2=st.columns([1,1])
            col1.markdown("<h3 style='color: #4b5563; margin-top: 15px; margin-bottom: 5px; font-weight: 600;'>📋 Description</h3>", unsafe_allow_html=True)
            col1.markdown(descbox(df['Desc'].values[0]), unsafe_allow_html=True)
            # Display symptoms           
            # Display prevention measures
            col2.markdown("<h3 style='color: #4b5563; margin-top: 15px; margin-bottom: 5px; font-weight: 600;'>🛡️ Prevention Measures</h3>", unsafe_allow_html=True)
            col2.markdown(prevbox(df['Prev'].values[0]), unsafe_allow_html=True)


elif select=='Blood Disease':
    # Background removed for professional UI
    st.markdown("<h1 style='text-align: center;color: #f00;'>Blood Diseases Detection</h1>", unsafe_allow_html=True)
    import numpy as np
    blood_model = keras.models.load_model('blood_model.h5', compile=False)
    blood_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    def predict_image(image_path):
        # Load the image
        img = image.load_img(image_path, target_size=(224, 224))
        
        # Preprocess the image (same preprocessing used in the ImageDataGenerator)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Make the image batch-like
        
        # MobileNetV2 preprocessing (used in ImageDataGenerator)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        # Make prediction
        prediction = blood_model.predict(img_array)
        
        # Get the class with the highest probability
        predicted_class = np.argmax(prediction, axis=1)
        
        # Class labels (same order as used during training)
        class_labels = ['Benign', 'Malignant_Pre-B', 'Malignant_Pro-B', 'Malignant_early Pre-B']
        
        return class_labels[predicted_class[0]], prediction
    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        if not is_blood_image(uploaded_file):
            st.error("Invalid input. Please upload a valid microscopic blood cell image.")
        else:
            # Display the uploaded image
            
            # Save the uploaded file temporarily
            with open("temp_image.jpg", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Predict the class of the uploaded image
            predicted_class, prediction = predict_image("temp_image.jpg")
            
            # Display the results in a box of html
            #open report.csv file
            rp = pd.read_csv('report.csv')
            rp.loc[rp['DiseaseType'] == 'blood', 'Class'] = predicted_class
            rp.to_csv('report.csv', index=False)
            data = pd.read_csv('info.csv', encoding='utf-8')
            df = data[data['Name'] == predicted_class]
            col1,col2=st.columns([1,1])
            col1.image(uploaded_file, caption=type, use_column_width=True)
            col2.markdown(box(predicted_class), unsafe_allow_html=True)
            col2.markdown("<h3 style='color: #4b5563; margin-top: 15px; margin-bottom: 5px; font-weight: 600;'>🩺 Symptoms</h3>", unsafe_allow_html=True)
            col2.markdown(symbox(df['Sym'].values[0]), unsafe_allow_html=True)
            # Display description
            col1,col2=st.columns([1,1])
            col1.markdown("<h3 style='color: #4b5563; margin-top: 15px; margin-bottom: 5px; font-weight: 600;'>📋 Description</h3>", unsafe_allow_html=True)
            col1.markdown(descbox(df['Desc'].values[0]), unsafe_allow_html=True)
            # Display symptoms           
            # Display prevention measures
            col2.markdown("<h3 style='color: #4b5563; margin-top: 15px; margin-bottom: 5px; font-weight: 600;'>🛡️ Prevention Measures</h3>", unsafe_allow_html=True)
            col2.markdown(prevbox(df['Prev'].values[0]), unsafe_allow_html=True)

elif select=='Liver Disease':
    # Background removed for professional UI
    info = pd.read_csv('liver_data.csv')
    info['Albumin_and_Globulin_Ratio'].fillna(info['Albumin_and_Globulin_Ratio'].median(),inplace=True)
    # info.isna().sum() 
    from sklearn import tree
    from sklearn.model_selection import train_test_split

    # info.info() # info

    dt = tree.DecisionTreeClassifier()
    #  rename 
    info.rename(columns ={'Dataset':'Target'},inplace=True)
    X = info.drop('Target',axis=1)
    y = info['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    dt.fit(X_train,y_train)
    st.markdown("<h1 style='text-align: center;color: #7a068c;'>Liver Disease Detection</h1>", unsafe_allow_html=True)
    with st.form('Prediction'):
        col1,col2 = st.columns(2)
        sex=col1.selectbox('Select Gender', ['Male','Female'])
        if sex=='Male':
            Sex=1
        else:
            Sex=2
        age = col2.slider('Select Age', 0, 100, 25) 
        col1,col2 = st.columns(2)
        Total_Bilirubin = col1.number_input("Enter your Total_Bilirubin") # 3
        Direct_Bilirubin = col2.number_input("Enter your Direct_Bilirubin")# 4
        Alkaline_Phosphotase = col1.number_input("Enter your Alkaline_Phosphotase") # 5
        Alamine_Aminotransferase = col2.number_input("Enter your Alamine_Aminotransferase") # 6
        Aspartate_Aminotransferase = col1.number_input("Enter your Aspartate_Aminotransferase") # 7
        Total_Protiens = col2.number_input("Enter your Total_Protiens")# 8
        Albumin = col1.number_input("Enter your Albumin") # 9
        Albumin_and_Globulin_Rati = col2.number_input("Enter your Albumin_and_Globulin_Ratio") # 10 
        col1,col2,col3=st.columns([3.2,1,3])
        if col2.form_submit_button('Submit',type='primary'):
            results = dt.predict([[Sex,age,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase
                            ,Total_Protiens,Albumin,Albumin_and_Globulin_Rati]])
            final=results[0]            
            if final==1:
                predicted_class = 'Liver Disease'
            else:
                predicted_class = 'Normal Liver'
            #open report.csv file
            rp = pd.read_csv('report.csv')
            rp.loc[rp['DiseaseType'] == 'liver', 'Class'] = predicted_class
            rp.to_csv('report.csv', index=False)
            data = pd.read_csv('info.csv', encoding='utf-8')
            df = data[data['Name'] == predicted_class]
            if predicted_class == 'Normal Liver':
                col1,col2=st.columns([1,1])
                pic='https://tse3.mm.bing.net/th/id/OIP.cZdJlYTX3kv_H_dsL_JH_AHaHa?pid=Api&P=0&h=220'
                col1.image(pic, caption='liver', use_column_width=True)
                col2.markdown(box('Normal Liver'), unsafe_allow_html=True)
                col2.markdown("<h3 style='color: #4b5563; margin-top: 15px; margin-bottom: 5px; font-weight: 600;'>🩺 Symptoms</h3>", unsafe_allow_html=True)
                col2.markdown(symbox(df['Sym'].values[0]), unsafe_allow_html=True)
                # Display description
                col1,col2=st.columns([1,1])
                col1.markdown("<h3 style='color: #4b5563; margin-top: 15px; margin-bottom: 5px; font-weight: 600;'>📋 Description</h3>", unsafe_allow_html=True)
                col1.markdown(descbox(df['Desc'].values[0]), unsafe_allow_html=True)
                # Display symptoms           
                # Display prevention measures
                col2.markdown("<h3 style='color: #4b5563; margin-top: 15px; margin-bottom: 5px; font-weight: 600;'>🛡️ Prevention Measures</h3>", unsafe_allow_html=True)
                col2.markdown(prevbox(df['Prev'].values[0]), unsafe_allow_html=True)
            else:
                col1,col2=st.columns([1,1])
                pic='https://www.gadhikarclinic.com/wp-content/uploads/2019/11/liver-image.jpg'
                col1.image(pic, caption='liver', use_column_width=True)
                col2.markdown(box('Liver Disease'), unsafe_allow_html=True)
                col2.markdown("<h3 style='color: #4b5563; margin-top: 15px; margin-bottom: 5px; font-weight: 600;'>🩺 Symptoms</h3>", unsafe_allow_html=True)
                col2.markdown(symbox(df['Sym'].values[0]), unsafe_allow_html=True)
                # Display description
                col1,col2=st.columns([1,1])
                col1.markdown("<h3 style='color: #4b5563; margin-top: 15px; margin-bottom: 5px; font-weight: 600;'>📋 Description</h3>", unsafe_allow_html=True)
                col1.markdown(descbox(df['Desc'].values[0]), unsafe_allow_html=True)
                # Display symptoms           
                # Display prevention measures
                col2.markdown("<h3 style='color: #4b5563; margin-top: 15px; margin-bottom: 5px; font-weight: 600;'>🛡️ Prevention Measures</h3>", unsafe_allow_html=True)
                col2.markdown(prevbox(df['Prev'].values[0]), unsafe_allow_html=True)

elif select=='Report':
    st.markdown("<h1 style='text-align: center;color: #f00;'>Clinical Diagnosis Report</h1>", unsafe_allow_html=True)
    
    name = st.session_state.get('patient_name', '').strip()
    if name:
        st.markdown(f"<h3 style='text-align: center; color: #333;'>Patient: {name} (Age: {st.session_state.get('patient_age')}, {st.session_state.get('patient_gender')})</h3>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter Patient Details in the sidebar before generating the final report.")

    df=pd.read_csv('report.csv')
    # Define prevention measures and diet plans for different disease types and classes
    prevention_diet = {
        "brain": {
            "Mild Dementia": {
                "Prevention": "Regular mental stimulation, a balanced diet, and physical exercise.",
                "Diet": "Leafy greens, fish, nuts, and berries."
            },
            "Moderate Dementia": {
                "Prevention": "Structured routines, cognitive therapy, and social engagement.",
                "Diet": "Omega-3 rich foods, whole grains, and lean proteins."
            },
            "No Dementia": {
                "Prevention": "Healthy lifestyle, cognitive exercises, and regular checkups.",
                "Diet": "A Mediterranean diet with fruits, vegetables, and healthy fats."
            },
            "Very Mild Dementia": {
                "Prevention": "Early diagnosis, physical activities, and memory training.",
                "Diet": "Antioxidant-rich foods, green tea, and vitamin B12."
            }
        },
        "skin": {
            "Melanoma": {
                "Prevention": "Use sunscreen, avoid excessive UV exposure, and check for skin changes.",
                "Diet": "Tomatoes, carrots, green tea, and foods rich in antioxidants."
            },
            "Actinic Keratosis": {
                "Prevention": "Wear protective clothing, use SPF 30+ sunscreen.",
                "Diet": "Leafy greens, citrus fruits, and foods rich in vitamin C."
            },
            "Basal Cell Carcinoma": {
                "Prevention": "Regular skin checkups, avoid prolonged sun exposure.",
                "Diet": "Tomatoes, omega-3 rich foods, and green tea."
            }
        },
        "lung": {
            "Normal Lung": {
                "Prevention": "Avoid smoking, minimize air pollution exposure, and exercise regularly.",
                "Diet": "Leafy greens, berries, and foods high in antioxidants."
            },
            "Adenocarcinoma": {
                "Prevention": "Avoid smoking, stay physically active, and eat a nutrient-rich diet.",
                "Diet": "Cruciferous vegetables, garlic, and omega-3 fatty acids."
            }
        },
        "heart": {
            "Heart Disease": {
                "Prevention": "Regular exercise, healthy diet, and stress management.",
                "Diet": "Oats, salmon, nuts, olive oil, and vegetables."
            },
            "Normal Heart": {
                "Prevention": "Maintain a balanced lifestyle, avoid processed foods, and get regular checkups.",
                "Diet": "Lean proteins, whole grains, and fiber-rich foods."
            }
        },
        "blood": {
            "Benign": {
                "Prevention": "Healthy lifestyle, balanced diet, and regular medical screenings.",
                "Diet": "Iron-rich foods like spinach, lean meats, and beans."
            },
            "Malignant_Pre-B": {
                "Prevention": "Early detection, avoiding carcinogens, and maintaining immunity.",
                "Diet": "Leafy greens, lean proteins, and antioxidant-rich foods."
            }
        },
        "liver": {
            "Normal Liver": {
                "Prevention": "Limit alcohol, avoid processed foods, and stay hydrated.",
                "Diet": "Leafy greens, garlic, beets, and green tea."
            },
            "Liver Disease": {
                "Prevention": "Healthy eating, maintaining weight, and regular exercise.",
                "Diet": "Fruits, whole grains, and low-fat dairy."
            }
        }
    }

    # Generate the PDF report
    # Background removed for professional UI
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.add_page()
    pdf.image("image.jpeg", x=0, y=0, w=297, h=210)
    pdf.set_font("Times", "B", 45)
    pdf.cell(0, 10, "Clinical Diagnosis Report", ln=True, align="C")
    pdf.ln(5)

    # Add Patient Details inside the PDF
    pdf.set_font("Arial", "B", 16)
    p_name = st.session_state.get('patient_name', 'N/A').strip() or 'N/A'
    p_age = st.session_state.get('patient_age', 'N/A')
    p_gender = st.session_state.get('patient_gender', 'N/A')
    
    pdf.cell(0, 10, f"Patient Name: {p_name}", ln=True, align="C")
    pdf.set_font("Arial", "", 14)
    pdf.cell(0, 10, f"Age: {p_age} | Gender: {p_gender}", ln=True, align="C")
    pdf.ln(10)

    # Disease information
    pdf.set_font("Arial", "", 12)
    # Prevention and diet plan
    for _, row in df.iterrows():
        disease_type = row["DiseaseType"].lower()
        disease_class = row["Class"]

        if disease_type in prevention_diet and disease_class in prevention_diet[disease_type]:
            pdf.set_font("Arial", "B", 16)
            pdf.set_text_color(255, 0, 0)
            pdf.cell(0, 10, f"{disease_type.capitalize()} - {disease_class}", ln=True)
            pdf.set_font("Arial", "B", 12)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 8, "Prevention:", ln=True)
            pdf.set_font("Arial", "", 12)
            pdf.multi_cell(0, 8, prevention_diet[disease_type][disease_class]["Prevention"])
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "Diet Plan:", ln=True)
            pdf.set_font("Arial", "", 12)
            pdf.multi_cell(0, 8, prevention_diet[disease_type][disease_class]["Diet"])
            pdf.ln(10)

    # Save the PDF
    col1,col2,col3=st.columns([1,5,1])
    col2.table(df)
    pdf_output_path = "Medical_Disease_Report.pdf"
    pdf.output(pdf_output_path)
    with open("Medical_Disease_Report.pdf", "rb") as file:
        col1,col2,col3=st.columns([2,2,1])
        btn = col2.download_button(
            label="Download Report",
            data=file,
            file_name="Medical_Disease_Report.pdf",
            mime="application/pdf",
            type='primary'
        )


