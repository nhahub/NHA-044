import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mediapipe as mp
import joblib
from scipy.spatial import ConvexHull
import mlflow
import mlflow.sklearn
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Hand_Gesture_Recognition_Advanced")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

# Define your paths
train_folder_path = r'C:\Users\Access\PythonImp\project_DEPI\train'
val_folder_path = r'C:\Users\Access\PythonImp\project_DEPI\val_final'
test_folder_path = r'C:\Users\Access\PythonImp\project_DEPI\test_final'

# Load the basic model to get class mappings
try:
    basic_model_data = joblib.load('basic_enhanced_model.pkl')
    label_to_idx = basic_model_data['label_to_idx']
    idx_to_label = basic_model_data['idx_to_label']
    class_names = basic_model_data['class_names']
    print("Loaded class mappings from basic model")
except:
    print("Basic model not found. Please run basic_enhanced_model.py first")
    exit()

# YOUR ORIGINAL FEATURE EXTRACTION FUNCTIONS (UNCHANGED)
def extract_hand_landmarks(image_path, hands):
    """
    Extract hand landmarks from image using MediaPipe
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract all 21 landmarks (x, y, z coordinates)
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(landmarks)
        else:
            return None
            
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((point1.x - point2.x)**2 + 
                  (point1.y - point2.y)**2 + 
                  (point1.z - point2.z)**2)

def extract_enhanced_features(hand_landmarks):
    """
    Extract additional geometric features
    """
    enhanced_features = []
    
    # Convert list back to landmark points for calculations
    landmarks = []
    for i in range(0, len(hand_landmarks), 3):
        class Point:
            def __init__(self, x, y, z):
                self.x, self.y, self.z = x, y, z
        landmarks.append(Point(hand_landmarks[i], hand_landmarks[i+1], hand_landmarks[i+2]))
    
    # Finger lengths (thumb, index, middle, ring, pinky)
    finger_tips = [4, 8, 12, 16, 20]
    finger_mcps = [2, 5, 9, 13, 17]
    
    for tip, mcp in zip(finger_tips, finger_mcps):
        length = calculate_distance(landmarks[tip], landmarks[mcp])
        enhanced_features.append(length)
    
    # Palm size (distance between wrist and middle finger MCP)
    palm_size = calculate_distance(landmarks[0], landmarks[9])
    enhanced_features.append(palm_size)
    
    return enhanced_features

def extract_advanced_features(hand_landmarks):
    """
    Extract even more sophisticated hand geometry features
    """
    advanced_features = []
    
    # Convert list back to landmark points for calculations
    landmarks = []
    for i in range(0, len(hand_landmarks), 3):
        class Point:
            def __init__(self, x, y, z):
                self.x, self.y, self.z = x, y, z
        landmarks.append(Point(hand_landmarks[i], hand_landmarks[i+1], hand_landmarks[i+2]))

    # 1. Relative finger lengths (normalized by palm size)
    palm_size = calculate_distance(landmarks[0], landmarks[9])
    finger_tips = [4, 8, 12, 16, 20]
    finger_mcps = [2, 5, 9, 13, 17]

    for tip, mcp in zip(finger_tips, finger_mcps):
        length = calculate_distance(landmarks[tip], landmarks[mcp])
        relative_length = length / palm_size if palm_size > 0 else 0
        advanced_features.append(relative_length)

    # 2. Finger curvature (distance from tip to line between base joints)
    for tip, pip, mcp in [(8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)]:
        curvature = calculate_curvature(landmarks[tip], landmarks[pip], landmarks[mcp])
        advanced_features.append(curvature)

    # 3. Hand convexity features
    convexity_features = calculate_convexity_features(landmarks)
    advanced_features.extend(convexity_features)

    # 4. Inter-finger distances
    inter_finger_dists = calculate_inter_finger_distances(landmarks)
    advanced_features.extend(inter_finger_dists)

    # 5. Palm center to finger tip distances
    palm_center = calculate_palm_center(landmarks)
    for tip_idx in finger_tips:
        dist = calculate_distance(palm_center, landmarks[tip_idx])
        advanced_features.append(dist)

    return advanced_features

def calculate_curvature(tip, pip, mcp):
    """Calculate finger curvature"""
    # Vector from MCP to PIP and MCP to tip
    v1 = np.array([pip.x - mcp.x, pip.y - mcp.y])
    v2 = np.array([tip.x - mcp.x, tip.y - mcp.y])

    if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
        cross_product = np.cross(v1, v2)
        return cross_product / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return 0

def calculate_convexity_features(landmarks):
    """Calculate hand convexity and compactness"""
    points = np.array([[lm.x, lm.y] for lm in landmarks])

    # Convex hull area vs bounding box area ratio
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        rect_area = (np.max(points[:,0]) - np.min(points[:,0])) * (np.max(points[:,1]) - np.min(points[:,1]))
        convexity_ratio = hull_area / rect_area if rect_area > 0 else 0
    except:
        convexity_ratio = 0

    return [convexity_ratio]

def calculate_inter_finger_distances(landmarks):
    """Calculate distances between finger tips"""
    finger_tips = [4, 8, 12, 16, 20]
    distances = []

    for i in range(len(finger_tips)):
        for j in range(i+1, len(finger_tips)):
            dist = calculate_distance(landmarks[finger_tips[i]], landmarks[finger_tips[j]])
            distances.append(dist)

    return distances

def calculate_palm_center(landmarks):
    """Calculate approximate palm center"""
    # Use wrist and MCP joints
    palm_points = [landmarks[0], landmarks[5], landmarks[9], landmarks[13], landmarks[17]]
    avg_x = sum(lm.x for lm in palm_points) / len(palm_points)
    avg_y = sum(lm.y for lm in palm_points) / len(palm_points)
    avg_z = sum(lm.z for lm in palm_points) / len(palm_points)

    class Point:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    return Point(avg_x, avg_y, avg_z)

def create_advanced_dataset(folder_path, expected_features=None):
    """Create dataset with advanced features"""
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

    landmarks_list = []
    labels_list = []

    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"⚠️ Folder does not exist: {folder_path}")
        hands.close()
        return np.array([]), np.array([])

    processed_count = 0
    skipped_count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(root, file)
                parent_folder = os.path.basename(os.path.dirname(full_path))

                image = cv2.imread(full_path)
                if image is None:
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]

                    # Extract basic landmarks from MediaPipe object
                    basic_landmarks = []
                    for landmark in hand_landmarks.landmark:
                        basic_landmarks.extend([landmark.x, landmark.y, landmark.z])
                    
                    # Convert to array for feature extraction
                    basic_landmarks_array = np.array(basic_landmarks)

                    # Enhanced features
                    enhanced_features = extract_enhanced_features(basic_landmarks_array)

                    # Advanced features
                    advanced_features = extract_advanced_features(basic_landmarks_array)

                    # Combine all features
                    all_features = np.concatenate([basic_landmarks_array, enhanced_features, advanced_features])
                    
                    # Check if we have the expected number of features
                    if expected_features is not None and len(all_features) != expected_features:
                        skipped_count += 1
                        continue

                    landmarks_list.append(all_features)
                    labels_list.append(parent_folder)
                    processed_count += 1
                else:
                    skipped_count += 1

    hands.close()
    print(f"  Processed {processed_count} images, skipped {skipped_count} (no hands/different features) from {folder_path}")
    return np.array(landmarks_list), np.array(labels_list)

def create_ensemble_model(X_train, y_train):
    """Create ensemble of multiple classifiers - YOUR ORIGINAL FUNCTION"""

    # Individual classifiers
    from sklearn.svm import SVC
    svm_model = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        probability=True,
        class_weight='balanced',
        random_state=42
    )

    rf_model = RandomForestClassifier(
        n_estimators=200,  # Your original: 200
        max_depth=20,      # Your original: 20
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    mlp_model = MLPClassifier(
        hidden_layer_sizes=(128, 64),  # Your original: (128, 64)
        activation='relu',
        alpha=0.01,
        learning_rate='adaptive',
        max_iter=1000,      # Your original: 1000
        random_state=42
    )

    # Voting classifier (soft voting)
    ensemble = VotingClassifier(
        estimators=[
            ('svm', svm_model),
            ('rf', rf_model),
            ('mlp', mlp_model)
        ],
        voting='soft',
        n_jobs=-1
    )

    return ensemble

def select_best_features(X_train, y_train, X_val, X_test, k=100):
    """Select most important features - YOUR ORIGINAL FUNCTION"""
    selector = SelectKBest(score_func=f_classif, k=min(k, X_train.shape[1]))
    X_train_selected = selector.fit_transform(X_train, y_train)
    
    # Transform validation and test sets only if they have the same feature dimension
    X_val_selected = np.array([])
    X_test_selected = np.array([])
    
    if len(X_val) > 0 and X_val.shape[1] == X_train.shape[1]:
        X_val_selected = selector.transform(X_val)
    elif len(X_val) > 0:
        print(f"⚠️ Validation set has {X_val.shape[1]} features, expected {X_train.shape[1]}. Skipping validation.")
        
    if len(X_test) > 0 and X_test.shape[1] == X_train.shape[1]:
        X_test_selected = selector.transform(X_test)
    elif len(X_test) > 0:
        print(f"⚠️ Test set has {X_test.shape[1]} features, expected {X_train.shape[1]}. Skipping test.")

    print(f"Selected {X_train_selected.shape[1]} best features from {X_train.shape[1]} total")
    return X_train_selected, X_val_selected, X_test_selected, selector

def analyze_confusion(actual, predicted, class_names):
    """Analyze which classes are most confused"""
    cm = confusion_matrix(actual, predicted)

    confusion_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append((class_names[i], class_names[j], cm[i, j]))

    # Sort by frequency of confusion
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)

    print("\nMost confused class pairs:")
    for true_class, pred_class, count in confusion_pairs[:10]:
        print(f"{true_class} → {pred_class}: {count} times")

def augment_weak_classes(X_data, y_data, class_names, weak_classes, augmentation_factor=2):
    """Augment data for weak-performing classes"""
    X_augmented = list(X_data)
    y_augmented = list(y_data)

    for class_name in weak_classes:
        if class_name not in label_to_idx:
            continue

        class_idx = label_to_idx[class_name]
        class_mask = (y_data == class_idx)
        class_samples = X_data[class_mask]

        if len(class_samples) == 0:
            continue

        # Add slightly perturbed versions
        for _ in range(augmentation_factor):
            for sample in class_samples:
                # Add small random noise
                noise = np.random.normal(0, 0.01, sample.shape)
                augmented_sample = sample + noise
                X_augmented.append(augmented_sample)
                y_augmented.append(class_idx)

    return np.array(X_augmented), np.array(y_augmented)

def log_confusion_matrix_plot(y_true, y_pred, class_names, filename):
    """Create and save confusion matrix plot"""
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def log_ensemble_parameters(ensemble_model):
    """Log ensemble model parameters to MLflow"""
    # Log Random Forest parameters
    try:
        rf_params = ensemble_model.estimators_[1].get_params()
        mlflow.log_params({
            "rf_n_estimators": rf_params.get('n_estimators', 200),
            "rf_max_depth": rf_params.get('max_depth', 20),
            "rf_min_samples_split": rf_params.get('min_samples_split', 5)
        })
    except:
        pass
    
    # Log SVM parameters
    try:
        svm_params = ensemble_model.estimators_[0].get_params()
        mlflow.log_params({
            "svm_C": svm_params.get('C', 10),
            "svm_kernel": svm_params.get('kernel', 'rbf'),
            "svm_gamma": svm_params.get('gamma', 'scale')
        })
    except:
        pass
    
    # Log MLP parameters
    try:
        mlp_params = ensemble_model.estimators_[2].get_params()
        mlflow.log_params({
            "mlp_hidden_layer_sizes": str(mlp_params.get('hidden_layer_sizes', (128, 64))),
            "mlp_activation": mlp_params.get('activation', 'relu'),
            "mlp_alpha": mlp_params.get('alpha', 0.01),
            "mlp_max_iter": mlp_params.get('max_iter', 1000)
        })
    except:
        pass

# MAIN EXECUTION WITH MLFLOW
print("="*50)
print("ADVANCED FEATURES AND ENSEMBLE MODEL WITH MLFLOW")
print("="*50)

# Start MLflow run
with mlflow.start_run(run_name=f"advanced_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    
    # Log experiment metadata
    mlflow.log_param("experiment_type", "hand_gesture_recognition")
    mlflow.log_param("feature_type", "advanced_geometric_ensemble")
    mlflow.log_param("model_architecture", "voting_ensemble_svm_rf_mlp")
    
    # Extract features from training set first to get expected feature dimension
    print("Extracting advanced features from training set...")
    X_train_advanced, y_train_advanced = create_advanced_dataset(train_folder_path)

    if len(X_train_advanced) == 0:
        raise ValueError("No hand landmarks were extracted from training data!")

    expected_features = X_train_advanced.shape[1]
    mlflow.log_param("expected_features", expected_features)

    # Now extract validation and test sets with the same expected features
    print("Processing validation set...")
    X_val_advanced, y_val_advanced = create_advanced_dataset(val_folder_path, expected_features=expected_features)

    print("Processing test set...")
    X_test_advanced, y_test_advanced = create_advanced_dataset(test_folder_path, expected_features=expected_features)

    print(f"\nDataset sizes:")
    print(f"Training: {X_train_advanced.shape[0]} samples")
    print(f"Validation: {X_val_advanced.shape[0] if len(X_val_advanced) > 0 else 0} samples")
    print(f"Test: {X_test_advanced.shape[0] if len(X_test_advanced) > 0 else 0} samples")
    print(f"Advanced feature dimension: {X_train_advanced.shape[1]}")

    # Log dataset statistics
    mlflow.log_params({
        "training_samples": X_train_advanced.shape[0],
        "validation_samples": X_val_advanced.shape[0] if len(X_val_advanced) > 0 else 0,
        "test_samples": X_test_advanced.shape[0] if len(X_test_advanced) > 0 else 0,
        "total_features": X_train_advanced.shape[1],
        "num_classes": len(class_names)
    })

    # Encode labels
    y_train_advanced_encoded = np.array([label_to_idx[label] for label in y_train_advanced])
    y_val_advanced_encoded = np.array([label_to_idx[label] for label in y_val_advanced]) if len(y_val_advanced) > 0 else np.array([])
    y_test_advanced_encoded = np.array([label_to_idx[label] for label in y_test_advanced]) if len(y_test_advanced) > 0 else np.array([])

    # Scale features
    scaler_advanced = StandardScaler()
    X_train_scaled_advanced = scaler_advanced.fit_transform(X_train_advanced)

    # Scale validation and test sets
    if len(X_val_advanced) > 0:
        X_val_scaled_advanced = scaler_advanced.transform(X_val_advanced)
    else:
        X_val_scaled_advanced = np.array([])

    if len(X_test_advanced) > 0:
        X_test_scaled_advanced = scaler_advanced.transform(X_test_advanced)
    else:
        X_test_scaled_advanced = np.array([])

    # Feature selection - YOUR ORIGINAL k=80
    print("Performing feature selection...")
    X_train_selected, X_val_selected, X_test_selected, feature_selector = select_best_features(
        X_train_scaled_advanced, y_train_advanced_encoded, X_val_scaled_advanced, X_test_scaled_advanced, k=80
    )
    
    mlflow.log_param("selected_features", X_train_selected.shape[1])
    mlflow.log_param("feature_selection_k", 80)

    # Train ensemble model - YOUR ORIGINAL ENSEMBLE
    print("Training ensemble model...")
    ensemble_model = create_ensemble_model(X_train_selected, y_train_advanced_encoded)
    ensemble_model.fit(X_train_selected, y_train_advanced_encoded)
    
    # Log ensemble parameters
    log_ensemble_parameters(ensemble_model)

    # Evaluate on validation set if available
    if len(X_val_selected) > 0 and len(y_val_advanced_encoded) > 0:
        y_val_pred_ensemble = ensemble_model.predict(X_val_selected)
        val_accuracy_ensemble = accuracy_score(y_val_advanced_encoded, y_val_pred_ensemble)
        print(f"Ensemble Validation Accuracy: {val_accuracy_ensemble:.4f}")
        mlflow.log_metric("ensemble_val_accuracy", val_accuracy_ensemble)

    # Evaluate on test set
    if len(X_test_selected) > 0 and len(y_test_advanced_encoded) > 0:
        y_test_pred_ensemble = ensemble_model.predict(X_test_selected)
        test_accuracy_ensemble = accuracy_score(y_test_advanced_encoded, y_test_pred_ensemble)
        print(f"Ensemble Test Accuracy: {test_accuracy_ensemble:.4f}")
        mlflow.log_metric("ensemble_test_accuracy", test_accuracy_ensemble)
        
        # Log detailed classification report
        test_report = classification_report(y_test_advanced_encoded, y_test_pred_ensemble, 
                                          target_names=class_names, output_dict=True)
        
        mlflow.log_metric("test_accuracy", test_report['accuracy'])
        mlflow.log_metric("test_macro_avg_f1", test_report['macro avg']['f1-score'])
        mlflow.log_metric("test_weighted_avg_f1", test_report['weighted avg']['f1-score'])
        
        # Log per-class metrics
        for class_name in class_names:
            if class_name in test_report:
                mlflow.log_metric(f"test_{class_name}_precision", test_report[class_name]['precision'])
                mlflow.log_metric(f"test_{class_name}_recall", test_report[class_name]['recall'])
                mlflow.log_metric(f"test_{class_name}_f1", test_report[class_name]['f1-score'])
        
        # Log confusion matrix as artifact
        log_confusion_matrix_plot(y_test_advanced_encoded, y_test_pred_ensemble, class_names, "confusion_matrix_ensemble.png")
        mlflow.log_artifact("confusion_matrix_ensemble.png")
        
        print("\nEnsemble Classification Report:")
        print(classification_report(y_test_advanced_encoded, y_test_pred_ensemble, target_names=class_names))
        
        # Analyze confusion
        analyze_confusion(y_test_advanced_encoded, y_test_pred_ensemble, class_names)

    # DATA AUGMENTATION FOR WEAK CLASSES - YOUR ORIGINAL AUGMENTATION
    weak_classes = ['seen', 'kaaf', 'fa', 'taa', 'toot']
    print(f"\nAugmenting weak classes: {weak_classes}")
    
    mlflow.log_param("augmented_classes", weak_classes)
    mlflow.log_param("augmentation_factor", 2)

    X_train_augmented, y_train_augmented = augment_weak_classes(
        X_train_selected, y_train_advanced_encoded, class_names, weak_classes
    )

    mlflow.log_metric("augmented_train_samples", X_train_augmented.shape[0])
    print(f"Training data after augmentation: {X_train_augmented.shape[0]} samples")

    # Final model with augmented data - YOUR ORIGINAL FINAL MODEL
    print("Training final model with augmented data...")
    final_model = create_ensemble_model(X_train_augmented, y_train_augmented)
    final_model.fit(X_train_augmented, y_train_augmented)
    
    # Log final model parameters
    log_ensemble_parameters(final_model)
    mlflow.log_param("model_type", "final_augmented_ensemble")

    # Evaluate final model on validation set
    if len(X_val_selected) > 0 and len(y_val_advanced_encoded) > 0:
        y_val_pred_final = final_model.predict(X_val_selected)
        val_accuracy_final = accuracy_score(y_val_advanced_encoded, y_val_pred_final)
        print(f"Final Model Validation Accuracy: {val_accuracy_final:.4f}")
        mlflow.log_metric("final_val_accuracy", val_accuracy_final)

    # Evaluate final model on test set
    if len(X_test_selected) > 0 and len(y_test_advanced_encoded) > 0:
        y_test_pred_final = final_model.predict(X_test_selected)
        test_accuracy_final = accuracy_score(y_test_advanced_encoded, y_test_pred_final)
        print(f"Final Model Test Accuracy: {test_accuracy_final:.4f}")
        mlflow.log_metric("final_test_accuracy", test_accuracy_final)
        
        # Log final model detailed metrics
        final_report = classification_report(y_test_advanced_encoded, y_test_pred_final, 
                                           target_names=class_names, output_dict=True)
        
        mlflow.log_metric("final_accuracy", final_report['accuracy'])
        mlflow.log_metric("final_macro_avg_f1", final_report['macro avg']['f1-score'])
        mlflow.log_metric("final_weighted_avg_f1", final_report['weighted avg']['f1-score'])
        
        # Log final per-class metrics
        for class_name in class_names:
            if class_name in final_report:
                mlflow.log_metric(f"final_{class_name}_precision", final_report[class_name]['precision'])
                mlflow.log_metric(f"final_{class_name}_recall", final_report[class_name]['recall'])
                mlflow.log_metric(f"final_{class_name}_f1", final_report[class_name]['f1-score'])
        
        # Log final confusion matrix
        log_confusion_matrix_plot(y_test_advanced_encoded, y_test_pred_final, class_names, "confusion_matrix_final.png")
        mlflow.log_artifact("confusion_matrix_final.png")

    # Save and log the final model
    final_model_data = {
        'ensemble_model': final_model,
        'scaler': scaler_advanced,
        'feature_selector': feature_selector,
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label,
        'class_names': class_names
    }

    # Log the model to MLflow
    mlflow.sklearn.log_model(
        final_model, 
        "final_ensemble_model",
        registered_model_name="HandGestureClassifier"
    )
    
    # Also save locally
    joblib.dump(final_model_data, 'final_ensemble_hand_model.pkl')
    mlflow.log_artifact('final_ensemble_hand_model.pkl')
    print("\nFinal ensemble model saved as 'final_ensemble_hand_model.pkl'")

    # Log overall experiment summary
    mlflow.log_metrics({
        "overall_test_accuracy": test_accuracy_final if 'test_accuracy_final' in locals() else 0,
        "total_features": X_train_advanced.shape[1],
        "selected_features_count": X_train_selected.shape[1],
        "total_classes": len(class_names)
    })
    
    # Log tags
    mlflow.set_tag("status", "completed")
    mlflow.set_tag("model_type", "advanced_ensemble")
    mlflow.set_tag("augmentation", "applied")

print("\n" + "="*50)
print("ADVANCED MODEL TRAINING COMPLETED WITH MLFLOW TRACKING!")
print("="*50)
if len(X_val_selected) > 0:
    print(f"Final Validation Accuracy: {val_accuracy_final:.4f}")
if len(X_test_selected) > 0:
    print(f"Final Test Accuracy: {test_accuracy_final:.4f}")
print(f"Number of advanced features: {X_train_advanced.shape[1]}")
print(f"Number of selected features: {X_train_selected.shape[1]}")
print(f"Number of classes: {len(class_names)}")
print(f"View your results at: http://localhost:5000")