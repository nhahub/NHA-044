import cv2
import numpy as np
import os
from skimage import measure, feature, filters
import matplotlib.pyplot as plt

input_dir = "C://Users//Access//Documents//data//data//sign_data//cleaned_grabcut_improved_test"
output_dir = "C://Users//Access//Documents//data//data//sign_data//filtered_improved_test"
os.makedirs(output_dir, exist_ok=True)

def analyze_image_quality(img):
    """Comprehensive image quality analysis"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 1. White pixel ratio analysis
    white_pixels = np.sum(gray > 240)
    black_pixels = np.sum(gray < 15)
    total_pixels = gray.size
    
    white_ratio = white_pixels / total_pixels
    non_white_ratio = 1 - white_ratio
    
    # 2. Edge detection for hand structure
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / total_pixels
    
    # 3. Contrast analysis
    contrast = gray.std()
    
    # 4. Blur detection (variance of Laplacian)
    blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 5. Object size and position analysis
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    object_area = 0
    object_centroid = None
    aspect_ratio = 0
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        object_area = cv2.contourArea(largest_contour)
        
        # Calculate centroid
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            object_centroid = (cx, cy)
        
        # Aspect ratio
        x, y, w_obj, h_obj = cv2.boundingRect(largest_contour)
        aspect_ratio = max(w_obj, h_obj) / min(w_obj, h_obj) if min(w_obj, h_obj) > 0 else 0
    
    # 6. Color distribution in non-white areas
    non_white_mask = gray < 240
    if np.any(non_white_mask):
        non_white_pixels = gray[non_white_mask]
        color_std = non_white_pixels.std()
        color_mean = non_white_pixels.mean()
    else:
        color_std = 0
        color_mean = 255
    
    return {
        'white_ratio': white_ratio,
        'non_white_ratio': non_white_ratio,
        'edge_density': edge_density,
        'contrast': contrast,
        'blur': blur_value,
        'object_area': object_area,
        'object_area_ratio': object_area / total_pixels,
        'aspect_ratio': aspect_ratio,
        'centroid': object_centroid,
        'color_std': color_std,
        'color_mean': color_mean,
        'width': w,
        'height': h
    }

def is_valid_hand_image(metrics, filename):
    """Determine if image contains a valid hand sign"""
    # 1. Reject images that are mostly white (no hand)
    if metrics['non_white_ratio'] < 0.02:
        return False, "Too white - no hand detected"
    
    # 2. Reject images that are mostly non-white (bad background removal)
    if metrics['non_white_ratio'] > 0.95:
        return False, "Poor background removal"
    
    # 3. Reject blurry images
    if metrics['blur'] < 50:  # Lower threshold for more strict blur detection
        return False, f"Too blurry ({metrics['blur']:.1f})"
    
    # 4. Reject low contrast images
    if metrics['contrast'] < 25:
        return False, f"Low contrast ({metrics['contrast']:.1f})"
    
    # 5. Object size constraints (hand should be reasonably sized)
    if metrics['object_area_ratio'] < 0.05:  # Object too small
        return False, f"Object too small ({metrics['object_area_ratio']:.3f})"
    
    if metrics['object_area_ratio'] > 0.95:  # Object too large (fills frame)
        return False, f"Object too large ({metrics['object_area_ratio']:.3f})"
    
    # 6. Aspect ratio constraints (hands are typically not extremely elongated)
    if metrics['aspect_ratio'] > 4.0:
        return False, f"Unnatural aspect ratio ({metrics['aspect_ratio']:.2f})"
    
    # 7. Edge density (hands should have reasonable edge complexity)
    if metrics['edge_density'] < 0.01:  # Too few edges
        return False, f"Too few edges ({metrics['edge_density']:.4f})"
    
    if metrics['edge_density'] > 0.3:   # Too many edges (noise)
        return False, f"Too many edges ({metrics['edge_density']:.4f})"
    
    # 8. Color consistency in non-white areas
    if metrics['color_std'] < 10:  # Too uniform color (might be artificial)
        return False, f"Too uniform color ({metrics['color_std']:.1f})"
    
    # 9. Centroid position (hand should be reasonably centered)
    if metrics['centroid']:
        cx, cy = metrics['centroid']
        center_x, center_y = metrics['width'] // 2, metrics['height'] // 2
        
        # Distance from center (normalized)
        dist_x = abs(cx - center_x) / center_x
        dist_y = abs(cy - center_y) / center_y
        
        if dist_x > 0.6 or dist_y > 0.6:  # Too far from center
            return False, f"Poor centering (dx:{dist_x:.2f}, dy:{dist_y:.2f})"
    
    # 10. Image size constraints
    if metrics['width'] < 100 or metrics['height'] < 100:
        return False, f"Image too small ({metrics['width']}x{metrics['height']})"
    
    return True, "Valid hand image"

def visualize_filtering(img, metrics, is_valid, reason, output_path=None):
    """Create visualization for debugging"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    plt.figure(figsize=(15, 4))
    
    # Original image
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"{'✅' if is_valid else '❌'} Original\n{reason}")
    plt.axis('off')
    
    # Grayscale
    plt.subplot(1, 4, 2)
    plt.imshow(gray, cmap='gray')
    plt.title(f'Grayscale\nWhite: {metrics["white_ratio"]:.3f}')
    plt.axis('off')
    
    # Edges
    plt.subplot(1, 4, 3)
    edges = cv2.Canny(gray, 50, 150)
    plt.imshow(edges, cmap='gray')
    plt.title(f'Edges\nDensity: {metrics["edge_density"]:.4f}')
    plt.axis('off')
    
    # Binary with centroid
    plt.subplot(1, 4, 4)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    plt.imshow(binary, cmap='gray')
    if metrics['centroid']:
        cx, cy = metrics['centroid']
        plt.plot(cx, cy, 'ro', markersize=8)
        plt.title(f'Object\nArea: {metrics["object_area_ratio"]:.3f}')
    else:
        plt.title('No object detected')
    plt.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

# Main processing for multiple subfolders
print("🔄 Starting improved image filtering...")
print("=" * 60)

# Get all subfolders
subfolders = [f.name for f in os.scandir(input_dir) if f.is_dir()]
print(f"📁 Found {len(subfolders)} subfolders: {subfolders}")

total_kept = 0
total_removed = 0
overall_removal_reasons = {}

# Create main debug directory
debug_dir = os.path.join(output_dir, "_debug_rejected")
os.makedirs(debug_dir, exist_ok=True)

# Process each subfolder
for subfolder in subfolders:
    subfolder_input_path = os.path.join(input_dir, subfolder)
    subfolder_output_path = os.path.join(output_dir, subfolder)
    subfolder_debug_path = os.path.join(debug_dir, subfolder)
    
    # Create output and debug subfolders
    os.makedirs(subfolder_output_path, exist_ok=True)
    os.makedirs(subfolder_debug_path, exist_ok=True)
    
    print(f"\n📂 Processing subfolder: {subfolder}")
    print("-" * 40)
    
    subfolder_kept = 0
    subfolder_removed = 0
    subfolder_removal_reasons = {}
    
    # Process images in current subfolder
    for filename in os.listdir(subfolder_input_path):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(subfolder_input_path, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Analyze image
        metrics = analyze_image_quality(img)
        
        # Check if valid hand image
        is_valid, reason = is_valid_hand_image(metrics, filename)
        
        if is_valid:
            # Save valid image
            cv2.imwrite(os.path.join(subfolder_output_path, filename), img)
            subfolder_kept += 1
            total_kept += 1
            print(f"  ✅ KEPT: {filename}")
        else:
            # Track removal reasons
            subfolder_removal_reasons[reason] = subfolder_removal_reasons.get(reason, 0) + 1
            overall_removal_reasons[reason] = overall_removal_reasons.get(reason, 0) + 1
            subfolder_removed += 1
            total_removed += 1
            
            # Save debug visualization for rejected images
            debug_path = os.path.join(subfolder_debug_path, f"rejected_{filename}")
            visualize_filtering(img, metrics, False, reason, debug_path)
            
            print(f"  ❌ REMOVED: {filename} - {reason}")
    
    # Print subfolder summary
    print(f"  📊 Subfolder {subfolder}: {subfolder_kept} kept, {subfolder_removed} removed")
    
    # Print subfolder removal reasons
    if subfolder_removal_reasons:
        print(f"  📈 Removal reasons for {subfolder}:")
        for reason, count in sorted(subfolder_removal_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"     {count:3d} × {reason}")

print("=" * 60)
print(f"📊 OVERALL FILTERING RESULTS:")
print(f"   ✅ Total kept: {total_kept} images")
print(f"   ❌ Total removed: {total_removed} images")
print(f"   📁 Output directory: {output_dir}")

# Print overall removal reasons summary
if overall_removal_reasons:
    print(f"\n📈 OVERALL REMOVAL REASONS:")
    for reason, count in sorted(overall_removal_reasons.items(), key=lambda x: x[1], reverse=True):
        print(f"   {count:4d} × {reason}")

# Print per-folder statistics
print(f"\n📊 PER-FOLDER BREAKDOWN:")
for subfolder in subfolders:
    subfolder_input_path = os.path.join(input_dir, subfolder)
    subfolder_output_path = os.path.join(output_dir, subfolder)
    
    input_count = len([f for f in os.listdir(subfolder_input_path) 
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    output_count = len([f for f in os.listdir(subfolder_output_path) 
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    
    removal_count = input_count - output_count
    retention_rate = (output_count / input_count * 100) if input_count > 0 else 0
    
    print(f"   {subfolder:15s}: {output_count:3d}/{input_count:3d} kept ({retention_rate:5.1f}%) - {removal_count:3d} removed")

# Calculate overall statistics
total_input = total_kept + total_removed
overall_retention_rate = (total_kept / total_input * 100) if total_input > 0 else 0

print(f"\n🎯 OVERALL STATISTICS:")
print(f"   Total images processed: {total_input}")
print(f"   Overall retention rate: {overall_retention_rate:.1f}%")
print(f"   Removal rate: {100 - overall_retention_rate:.1f}%")

print(f"\n🎯 QUALITY THRESHOLDS USED:")
print(f"   • Non-white ratio: 0.02 < ratio < 0.95")
print(f"   • Blur threshold: > 50 (Laplacian variance)")
print(f"   • Contrast threshold: > 25")
print(f"   • Object size: 0.05 < ratio < 0.95")
print(f"   • Aspect ratio: < 4.0")
print(f"   • Edge density: 0.01 < density < 0.3")
print(f"   • Color variation: std > 10")

print(f"\n💡 TIPS:")
print(f"   • Check {debug_dir} for visualization of rejected images")
print(f"   • Adjust thresholds in is_valid_hand_image() if needed")
print(f"   • Manual review recommended for borderline cases")
print(f"   • Subfolder-specific analysis available in debug directories")