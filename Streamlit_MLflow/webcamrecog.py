import cv2
import numpy as np
import mediapipe as mp
import joblib
from scipy.spatial import ConvexHull

# Load your trained model
model_data = joblib.load('final_ensemble_hand_model.pkl')
model = model_data['ensemble_model']
scaler = model_data['scaler']
idx_to_label = model_data['idx_to_label']

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((point1.x - point2.x)**2 + 
                  (point1.y - point2.y)**2 + 
                  (point1.z - point2.z)**2)

def extract_enhanced_features(landmarks_list):
    """
    Extract enhanced geometric features (6 features)
    """
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
    """
    Extract advanced geometric features (25 features)
    """
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
    """
    Extract EXACTLY the same features as during training (94 features total)
    """
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

print(" Starting Arabic Hand Sign Recognition")
print(" Show your hand - Press 'q' to quit")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip for mirror view
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame
    results = hands.process(rgb_frame)
    
    prediction_text = "Show Hand"
    confidence = 0.0
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        try:
            # Extract EXACT features as training (94 features)
            features = extract_all_features(hand_landmarks)
            features = features.reshape(1, -1)
            
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Apply feature selection (if your model has it)
            if 'feature_selector' in model_data:
                features_processed = model_data['feature_selector'].transform(features_scaled)
            else:
                features_processed = features_scaled
            
            # PREDICT
            prediction = model.predict(features_processed)[0]
            confidence_scores = model.predict_proba(features_processed)
            confidence = np.max(confidence_scores)
            
            # Get class name
            prediction_text = idx_to_label[prediction]
            
            # Draw landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
        except Exception as e:
            prediction_text = f"Processing..."
    
    # Display prediction
    color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255) if confidence > 0 else (0, 0, 255)
    
    cv2.putText(frame, f"Prediction: {prediction_text}", (30, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    if confidence > 0:
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (30, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Show frame
    cv2.imshow('Arabic Hand Sign Recognition', frame)
    
    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print(" Done!")