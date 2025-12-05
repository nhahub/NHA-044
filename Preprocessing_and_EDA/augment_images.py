import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# المسارات
input_dir = "C://Users//Access//Documents//data//data//sign_data//filtered_improved_test"
output_dir = "C://Users//Access//PythonImp//project_DEPI//augmented_test"
os.makedirs(output_dir, exist_ok=True)

# إعدادات الـAugmentation
datagen = ImageDataGenerator(
    rotation_range=20,          # تدوير عشوائي ±20 درجة
    width_shift_range=0.1,      # تحريك أفقي بسيط
    height_shift_range=0.1,     # تحريك رأسي بسيط
    zoom_range=0.15,            # تكبير أو تصغير بسيط
    brightness_range=[0.8, 1.2],# تغيير في الإضاءة
    horizontal_flip=True,       # قلب أفقي
    fill_mode='nearest'
)

# عدد النسخ لكل صورة (تقدر تغيره حسب وقت المعالجة)
NUM_AUG_PER_IMAGE = 3

# إحصائيات
total_original_images = 0
total_augmented_images = 0

# معالجة كل مجلد فرعي
for subfolder in os.listdir(input_dir):
    subfolder_path = os.path.join(input_dir, subfolder)
    output_subfolder_path = os.path.join(output_dir, subfolder)
    
    # التأكد من أن العنصر مجلد وليس ملف
    if not os.path.isdir(subfolder_path):
        continue
    
    # إنشاء المجلد الفرعي في المجلد الهدف
    os.makedirs(output_subfolder_path, exist_ok=True)
    
    print(f"🔍 معالجة المجلد: {subfolder}")
    
    subfolder_count = 0
    for filename in os.listdir(subfolder_path):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        input_path = os.path.join(subfolder_path, filename)
        img = cv2.imread(input_path)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, 0)

        # حفظ الصورة الأصلية في المجلد الجديد
        original_output_path = os.path.join(output_subfolder_path, f"original_{filename}")
        cv2.imwrite(original_output_path, cv2.cvtColor(img[0], cv2.COLOR_RGB2BGR))
        
        # إنشاء الصور المحسنة
        i = 0
        for batch in datagen.flow(img, batch_size=1,
                                  save_to_dir=output_subfolder_path,
                                  save_prefix=f"aug_{filename.split('.')[0]}",
                                  save_format='jpg'):
            i += 1
            if i >= NUM_AUG_PER_IMAGE:
                break
        
        subfolder_count += 1
        total_original_images += 1
        total_augmented_images += NUM_AUG_PER_IMAGE
    
    print(f"   ✅ تم معالجة {subfolder_count} صورة في المجلد {subfolder}")

print(f"\n🎉 تم إنشاء صور Augmentation جديدة بنجاح!")
print(f"📊 الإحصائيات النهائية:")
print(f"   📸 عدد الصور الأصلية: {total_original_images}")
print(f"   🖼️  عدد الصور المحسنة: {total_augmented_images}")
print(f"   📈 العدد الإجمالي: {total_original_images + total_augmented_images}")
print(f"   📁 الصور الجديدة محفوظة في: {output_dir}")