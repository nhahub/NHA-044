import os
import cv2

# المسارات
input_dir = "C://Users//Access//Documents//data//data//sign_data//images"
output_dir = "C://Users//Access//Documents//data//data//sign_data//processed"

# تأكد إن المجلد موجود
os.makedirs(output_dir, exist_ok=True)

# نقرأ كل الصور
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".jpg"):
        path = os.path.join(input_dir, filename)
        img = cv2.imread(path)

        if img is None:
            print(f"❌ الصورة فيها مشكلة: {filename}")
            continue

        # 1️⃣ نغير الحجم
        resized = cv2.resize(img, (128, 128))

        # 2️⃣ نطبّع القيم (normalize)
        normalized = resized / 255.0

        # 3️⃣ نحفظ الصورة بعد المعالجة
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, (normalized * 255))

print("✅ تم معالجة جميع الصور وحفظها في مجلد processed/")