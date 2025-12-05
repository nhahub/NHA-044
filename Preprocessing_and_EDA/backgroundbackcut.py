import cv2
import numpy as np
import os
from skimage import filters, segmentation, morphology
import matplotlib.pyplot as plt

input_dir = "C://Users//Access//PythonImp//project_DEPI//test_final"
output_dir = "C://Users//Access//Documents//data//data//sign_data//cleaned_grabcut_improved_test"
os.makedirs(output_dir, exist_ok=True)

def preprocess_image(img):
    """Preprocess image to improve GrabCut results"""
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Enhance contrast using CLAHE
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(lab))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return enhanced

def create_improved_mask(img):
    """Create an improved initial mask using multiple techniques"""
    h, w = img.shape[:2]
    
    # Method 1: Simple center rectangle (original approach)
    rect_mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(rect_mask, (int(w*0.1), int(h*0.1)), (int(w*0.9), int(h*0.9)), 1, -1)
    
    # Method 2: Color-based segmentation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Common skin color ranges in HSV
    lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
    lower_skin2 = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin2 = np.array([20, 255, 255], dtype=np.uint8)
    
    skin_mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
    skin_mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
    skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
    
    # Clean up the skin mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    
    # Convert to binary mask
    skin_mask_binary = (skin_mask > 0).astype(np.uint8)
    
    # Combine both methods
    combined_mask = cv2.bitwise_or(rect_mask, skin_mask_binary)
    
    return combined_mask

def improved_grabcut(img, initial_mask):
    """Improved GrabCut with better initialization"""
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    # Initialize mask based on our improved initial mask
    mask[initial_mask == 1] = cv2.GC_PR_FGD  # Probably foreground
    mask[initial_mask == 0] = cv2.GC_PR_BGD  # Probably background
    
    # Run GrabCut with mask initialization
    cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    
    return mask

def post_process_mask(mask, img):
    """Post-process the mask to clean up edges and fill holes"""
    # Create final mask
    final_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # Remove small noise
    kernel = np.ones((3, 3), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Fill holes
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours and keep only the largest one (main object)
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        final_mask = np.zeros_like(final_mask)
        cv2.drawContours(final_mask, [largest_contour], -1, 1, -1)
    
    # Smooth edges
    final_mask = cv2.GaussianBlur(final_mask.astype(np.float32), (5, 5), 0)
    final_mask = (final_mask > 0.5).astype(np.uint8)
    
    return final_mask

def apply_white_background(img, mask):
    """Apply white background with smooth edges"""
    # Create result with transparent background first
    result = img * mask[:, :, np.newaxis]
    
    # Create smooth edges using Gaussian blur on mask
    smooth_mask = cv2.GaussianBlur(mask.astype(np.float32), (3, 3), 0)
    smooth_mask = smooth_mask[:, :, np.newaxis]
    
    # Blend with white background
    white_bg = np.full_like(img, 255)
    cleaned = (result * smooth_mask + white_bg * (1 - smooth_mask)).astype(np.uint8)
    
    return cleaned

def process_single_image(img_path, output_path, debug=False):
    """Process a single image with improved GrabCut"""
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Cannot read image: {img_path}")
        return False
    
    original_img = img.copy()
    
    try:
        # Step 1: Preprocess image
        processed_img = preprocess_image(img)
        
        # Step 2: Create improved initial mask
        initial_mask = create_improved_mask(processed_img)
        
        # Step 3: Apply improved GrabCut
        grabcut_mask = improved_grabcut(processed_img, initial_mask)
        
        # Step 4: Post-process the mask
        final_mask = post_process_mask(grabcut_mask, processed_img)
        
        # Step 5: Apply white background
        result = apply_white_background(original_img, final_mask)
        
        # Save result
        cv2.imwrite(output_path, result)
        
        if debug:
            # Debug visualization
            plt.figure(figsize=(20, 4))
            
            plt.subplot(1, 5, 1)
            plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
            plt.title('Original')
            plt.axis('off')
            
            plt.subplot(1, 5, 2)
            plt.imshow(initial_mask, cmap='gray')
            plt.title('Initial Mask')
            plt.axis('off')
            
            plt.subplot(1, 5, 3)
            grabcut_vis = np.where((grabcut_mask == 2) | (grabcut_mask == 0), 0, 1)
            plt.imshow(grabcut_vis, cmap='gray')
            plt.title('GrabCut Result')
            plt.axis('off')
            
            plt.subplot(1, 5, 4)
            plt.imshow(final_mask, cmap='gray')
            plt.title('Final Mask')
            plt.axis('off')
            
            plt.subplot(1, 5, 5)
            plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            plt.title('Final Result')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{output_path}_debug.jpg", dpi=150, bbox_inches='tight')
            plt.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {img_path}: {str(e)}")
        return False

# Main processing loop for multiple subfolders
print("🔄 Starting improved background removal with GrabCut...")
print("=" * 60)

# Get all subfolders (assuming they are the 32 classes)
subfolders = [f.name for f in os.scandir(input_dir) if f.is_dir()]
print(f"📁 Found {len(subfolders)} subfolders: {subfolders}")

total_success_count = 0
total_images_count = 0

for subfolder in subfolders:
    subfolder_input_path = os.path.join(input_dir, subfolder)
    subfolder_output_path = os.path.join(output_dir, subfolder)
    
    # Create output subfolder
    os.makedirs(subfolder_output_path, exist_ok=True)
    
    print(f"\n📂 Processing subfolder: {subfolder}")
    print("-" * 40)
    
    subfolder_success_count = 0
    subfolder_total_count = 0
    
    # Process images in current subfolder
    for filename in os.listdir(subfolder_input_path):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        subfolder_total_count += 1
        total_images_count += 1
        
        img_path = os.path.join(subfolder_input_path, filename)
        output_path = os.path.join(subfolder_output_path, filename)
        
        print(f"  Processing {filename}...", end=" ")
        
        success = process_single_image(img_path, output_path, debug=False)
        
        if success:
            subfolder_success_count += 1
            total_success_count += 1
            print("✅")
        else:
            print("❌")
    
    # Print subfolder summary
    print(f"  📊 Subfolder {subfolder}: {subfolder_success_count}/{subfolder_total_count} images processed successfully")

print("=" * 60)
print(f"🎉 Background removal completed!")
print(f"📊 Overall Statistics:")
print(f"   - Total subfolders processed: {len(subfolders)}")
print(f"   - Total images processed: {total_images_count}")
print(f"   - Successfully processed: {total_success_count}")
print(f"   - Success rate: {(total_success_count/total_images_count)*100:.1f}%")
print(f"📁 Output directory: {output_dir}")

# Print per-folder summary
print(f"\n📈 Per-folder breakdown:")
for subfolder in subfolders:
    subfolder_input_path = os.path.join(input_dir, subfolder)
    subfolder_output_path = os.path.join(output_dir, subfolder)
    
    input_count = len([f for f in os.listdir(subfolder_input_path) 
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    output_count = len([f for f in os.listdir(subfolder_output_path) 
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    
    print(f"   {subfolder}: {output_count}/{input_count} images")

print(f"\n💡 Tips for difficult images:")
print("   - Manual rectangle adjustment")
print("   - Different color space thresholds") 
print("   - Multiple GrabCut iterations")
print("   - Check individual subfolder results for consistency")