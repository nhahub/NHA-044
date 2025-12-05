import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import joblib
from scipy.spatial import ConvexHull
import tempfile
import os
from PIL import Image
import time

# Set page configuration
st.set_page_config(
    page_title="Arabic Hand Sign Recognition",
    page_icon="✋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .arabic-letter {
        font-size: 4rem;
        text-align: center;
        color: #1f77b4;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #00ff00;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffa500;
        font-weight: bold;
    }
    .confidence-low {
        color: #ff0000;
        font-weight: bold;
    }
    .video-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Arabic letter mapping dictionary
label_map = {
    'ain': 'ع',
    'al': 'ال',
    'aleff': 'ا',
    'bb': 'ب',
    'dal': 'د',
    'dha': 'ظ',
    'dhad': 'ض',
    'fa': 'ف',
    'gaaf': 'ق',
    'ghain': 'غ',
    'ha': 'ه',
    'haa': 'ح',
    'jeem': 'ج',
    'kaaf': 'ك',
    'khaa': 'خ',
    'la': 'لا',
    'laam': 'ل',
    'meem': 'م',
    'nun': 'ن',
    'ra': 'ر',
    'saad': 'ص',
    'seen': 'س',
    'sheen': 'ش',
    'ta': 'ت',
    'taa': 'ط',
    'thaa': 'ث',
    'thal': 'ذ',
    'toot': 'ة',
    'waw': 'و',
    'ya': 'ى',
    'yaa': 'ي',
    'zay': 'ز'
}

# Initialize MediaPipe Hands
@st.cache_resource
def initialize_mediapipe():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    return mp_hands, hands

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model_data = joblib.load('final_ensemble_hand_model.pkl')
        return model_data
    except FileNotFoundError:
        st.error("Model file 'final_ensemble_hand_model.pkl' not found. Please make sure it's in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Helper functions (same as original)
def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((point1.x - point2.x)**2 + 
                  (point1.y - point2.y)**2 + 
                  (point1.z - point2.z)**2)

def extract_enhanced_features(landmarks_list):
    """Extract enhanced geometric features (6 features)"""
    enhanced_features = []
    
    # Convert list to landmark points
    landmarks = []
    for i in range(0, len(landmarks_list), 3):
        class Point:
            def __init__(self, x, y, z):
                self.x, self.y, self.z = x, y, z
        landmarks.append(Point(landmarks_list[i], landmarks_list[i+1], landmarks_list[i+2]))
    
    # Finger lengths (thumb, index, middle, ring, pinky)
    finger_tips = [4, 8, 12, 16, 20]
    finger_mcps = [2, 5, 9, 13, 17]
    
    for tip, mcp in zip(finger_tips, finger_mcps):
        length = calculate_distance(landmarks[tip], landmarks[mcp])
        enhanced_features.append(length)
    
    # Palm size
    palm_size = calculate_distance(landmarks[0], landmarks[9])
    enhanced_features.append(palm_size)
    
    return np.array(enhanced_features)

def extract_advanced_features(landmarks_list):
    """Extract advanced geometric features (25 features)"""
    advanced_features = []
    
    # Convert list to landmark points
    landmarks = []
    for i in range(0, len(landmarks_list), 3):
        class Point:
            def __init__(self, x, y, z):
                self.x, self.y, self.z = x, y, z
        landmarks.append(Point(landmarks_list[i], landmarks_list[i+1], landmarks_list[i+2]))

    # 1. Relative finger lengths
    palm_size = calculate_distance(landmarks[0], landmarks[9])
    finger_tips = [4, 8, 12, 16, 20]
    finger_mcps = [2, 5, 9, 13, 17]

    for tip, mcp in zip(finger_tips, finger_mcps):
        length = calculate_distance(landmarks[tip], landmarks[mcp])
        relative_length = length / palm_size if palm_size > 0 else 0
        advanced_features.append(relative_length)

    # 2. Finger curvature
    def calculate_curvature(tip, pip, mcp):
        v1 = np.array([pip.x - mcp.x, pip.y - mcp.y])
        v2 = np.array([tip.x - mcp.x, tip.y - mcp.y])
        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            cross_product = np.cross(v1, v2)
            return cross_product / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return 0

    for tip, pip, mcp in [(8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)]:
        curvature = calculate_curvature(landmarks[tip], landmarks[pip], landmarks[mcp])
        advanced_features.append(curvature)

    # 3. Hand convexity
    points = np.array([[lm.x, lm.y] for lm in landmarks])
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        rect_area = (np.max(points[:,0]) - np.min(points[:,0])) * (np.max(points[:,1]) - np.min(points[:,1]))
        convexity_ratio = hull_area / rect_area if rect_area > 0 else 0
    except:
        convexity_ratio = 0
    advanced_features.append(convexity_ratio)

    # 4. Inter-finger distances
    finger_tips = [4, 8, 12, 16, 20]
    for i in range(len(finger_tips)):
        for j in range(i+1, len(finger_tips)):
            dist = calculate_distance(landmarks[finger_tips[i]], landmarks[finger_tips[j]])
            advanced_features.append(dist)

    # 5. Palm center to finger tip distances
    palm_points = [landmarks[0], landmarks[5], landmarks[9], landmarks[13], landmarks[17]]
    avg_x = sum(lm.x for lm in palm_points) / len(palm_points)
    avg_y = sum(lm.y for lm in palm_points) / len(palm_points)
    avg_z = sum(lm.z for lm in palm_points) / len(palm_points)
    
    class Point: 
        def __init__(self, x, y, z): self.x, self.y, self.z = x, y, z
    palm_center = Point(avg_x, avg_y, avg_z)
    
    for tip_idx in finger_tips:
        dist = calculate_distance(palm_center, landmarks[tip_idx])
        advanced_features.append(dist)

    return np.array(advanced_features)

def extract_all_features(hand_landmarks):
    """Extract EXACTLY the same features as during training (94 features total)"""
    # 1. Basic landmarks (63 features)
    basic_landmarks = []
    for landmark in hand_landmarks.landmark:
        basic_landmarks.extend([landmark.x, landmark.y, landmark.z])
    basic_landmarks = np.array(basic_landmarks)
    
    # 2. Enhanced features (6 features)
    enhanced_features = extract_enhanced_features(basic_landmarks)
    
    # 3. Advanced features (25 features)
    advanced_features = extract_advanced_features(basic_landmarks)
    
    # 4. Combine ALL features (63 + 6 + 25 = 94 features)
    all_features = np.concatenate([basic_landmarks, enhanced_features, advanced_features])
    
    return all_features

def process_frame(frame, model_data, mp_hands, hands):
    """Process a single frame and return prediction results"""
    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame
    results = hands.process(rgb_frame)
    
    prediction_text = "No Hand Detected"
    confidence = 0.0
    processed_frame = frame.copy()
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        try:
            # Extract EXACT features as training (94 features)
            features = extract_all_features(hand_landmarks)
            features = features.reshape(1, -1)
            
            # Scale features
            features_scaled = model_data['scaler'].transform(features)
            
            # Apply feature selection (if your model has it)
            if 'feature_selector' in model_data:
                features_processed = model_data['feature_selector'].transform(features_scaled)
            else:
                features_processed = features_scaled
            
            # PREDICT
            prediction = model_data['ensemble_model'].predict(features_processed)[0]
            confidence_scores = model_data['ensemble_model'].predict_proba(features_processed)
            confidence = np.max(confidence_scores)
            
            # Get class name
            prediction_text = model_data['idx_to_label'][prediction]
            
            # Draw landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                processed_frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
        except Exception as e:
            prediction_text = f"Processing Error: {str(e)}"
    
    return processed_frame, prediction_text, confidence

def main():
    # Header
    st.markdown('<h1 class="main-header">✋ Arabic Hand Sign Recognition</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Mode", 
                                   ["Real-time Webcam", "Upload Image", "About"])
    
    # Load model and initialize MediaPipe
    model_data = load_model()
    if model_data is None:
        st.error("Failed to load model. Please check if 'final_ensemble_hand_model.pkl' exists.")
        return
    
    mp_hands, hands = initialize_mediapipe()
    
    if app_mode == "Real-time Webcam":
        real_time_mode(model_data, mp_hands, hands)
    elif app_mode == "Upload Image":
        upload_image_mode(model_data, mp_hands, hands)
    elif app_mode == "About":
        about_mode()

def real_time_mode(model_data, mp_hands, hands):
    st.markdown("### Real-time Hand Sign Recognition")
    st.write("Show your hand to the webcam for real-time Arabic sign language recognition.")
    
    # Webcam setup
    run_camera = st.checkbox("Start Webcam", value=False)
    FRAME_WINDOW = st.image([])
    
    # Confidence threshold
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.05)
    
    # Statistics
    if 'predictions' not in st.session_state:
        st.session_state.predictions = []
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Live Feed")
    
    with col2:
        st.markdown("#### Prediction Results")
        prediction_placeholder = st.empty()
        confidence_placeholder = st.empty()
        arabic_placeholder = st.empty()
        stats_placeholder = st.empty()
    
    if run_camera:
        cap = cv2.VideoCapture(0)
        
        while run_camera:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam")
                break
            
            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            
            # Process frame
            processed_frame, prediction, confidence = process_frame(frame, model_data, mp_hands, hands)
            
            # Update session state
            if confidence > confidence_threshold:
                st.session_state.predictions.append((prediction, confidence))
                if len(st.session_state.predictions) > 100:  # Keep last 100 predictions
                    st.session_state.predictions.pop(0)
            
            # Convert BGR to RGB for display
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(processed_frame_rgb)
            
            # Update prediction display
            with col2:
                # Get Arabic letter from label_map
                arabic_letter = label_map.get(prediction, prediction)
                
                # Display Arabic letter prominently
                if prediction != "No Hand Detected" and confidence > confidence_threshold:
                    arabic_placeholder.markdown(
                        f'<div class="arabic-letter">{arabic_letter}</div>', 
                        unsafe_allow_html=True
                    )
                else:
                    arabic_placeholder.empty()
                
                prediction_placeholder.markdown(
                    f"""
                    <div class="prediction-box">
                        <h3>Prediction: {prediction}</h3>
                        <p>Arabic Letter: <strong>{arabic_letter}</strong></p>
                        <p>Confidence: <span class="{
                            'confidence-high' if confidence > 0.8 
                            else 'confidence-medium' if confidence > 0.6 
                            else 'confidence-low'
                        }">{confidence:.2f}</span></p>
                    </div>
                    """, unsafe_allow_html=True
                )
                
                # Display statistics
                if st.session_state.predictions:
                    recent_preds = [p[0] for p in st.session_state.predictions[-20:]]  # Last 20 predictions
                    if recent_preds:
                        most_common = max(set(recent_preds), key=recent_preds.count)
                        arabic_common = label_map.get(most_common, most_common)
                        stats_placeholder.write(f"**Most Recent Common Sign:** {most_common} ({arabic_common})")
            
            # Check if user wants to stop
            if not st.session_state.get('run_camera', True):
                break
        
        cap.release()
    
    # Stop button
    if st.button("Stop Camera"):
        st.session_state.run_camera = False
        st.rerun()

def upload_image_mode(model_data, mp_hands, hands):
    st.markdown("### Upload Image for Hand Sign Recognition")
    st.write("Upload an image containing a hand gesture for classification.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Convert to OpenCV format
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Process image
        with st.spinner("Processing image..."):
            processed_frame, prediction, confidence = process_frame(frame, model_data, mp_hands, hands)
        
        # Convert back to RGB for display
        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        with col2:
            st.image(processed_frame_rgb, caption="Processed Image with Landmarks", use_column_width=True)
        
        # Get Arabic letter from label_map
        arabic_letter = label_map.get(prediction, prediction)
        
        # Display results
        st.markdown(
            f"""
            <div class="prediction-box">
                <h2>Recognition Result</h2>
                <div class="arabic-letter">{arabic_letter}</div>
                <h3>Predicted Sign: {prediction}</h3>
                <p>Arabic Letter: <strong>{arabic_letter}</strong></p>
                <p>Confidence: <span class="{
                    'confidence-high' if confidence > 0.8 
                    else 'confidence-medium' if confidence > 0.6 
                    else 'confidence-low'
                }">{confidence:.2f}</span></p>
            </div>
            """, unsafe_allow_html=True
        )

def about_mode():
    st.markdown("## About Arabic Hand Sign Recognition")
    
    st.markdown("""
    ### 🤖 How It Works
    
    This application uses advanced computer vision and machine learning to recognize Arabic sign language gestures in real-time.
    
    **Technology Stack:**
    - **MediaPipe**: For hand landmark detection
    - **Ensemble Machine Learning Model**: For gesture classification
    - **Streamlit**: For web deployment
    - **OpenCV**: For image processing
    
    **Feature Extraction:**
    - 63 basic hand landmark coordinates
    - 6 enhanced geometric features (finger lengths, palm size)
    - 25 advanced features (curvature, convexity, inter-finger distances)
    - **Total: 94 features** for robust classification
    
    ### 🎯 How to Use
    
    1. **Real-time Webcam Mode**: 
       - Click "Start Webcam" 
       - Show your hand to the camera
       - View real-time predictions with Arabic letter display
    
    2. **Upload Image Mode**:
       - Upload a clear image of a hand gesture
       - Get instant classification results with Arabic letter
    
    ### 📊 Model Information
    
    - **Model Type**: Ensemble classifier
    - **Training**: Trained on Arabic sign language dataset
    - **Features**: 94-dimensional feature vector
    - **Confidence**: Displayed for each prediction
    
    ### 👋 Supported Gestures
    
    The system recognizes various Arabic alphabet signs:
    """)
    
    # Display the label map in a nice format
    cols = st.columns(4)
    items_per_col = (len(label_map) + 3) // 4  # Divide into 4 columns
    
    for i, (eng, arabic) in enumerate(label_map.items()):
        col_idx = i // items_per_col
        with cols[col_idx]:
            st.write(f"**{eng}**: {arabic}")
    
    # Add some statistics or model info if available
    st.markdown("---")
    st.markdown("### Technical Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Feature Dimensions", "94")
    
    with col2:
        st.metric("Hand Landmarks", "21")
    
    with col3:
        st.metric("Processing", "Real-time")

if __name__ == "__main__":
    main()