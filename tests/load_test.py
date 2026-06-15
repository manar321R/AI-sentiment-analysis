import os
import json
import pandas as pd
import requests
import time
import kagglehub

# 1. تحميل وتقسيم الداتاسيت باستخدام kagglehub
def download_and_split_data():
    print("⏳ Downloading dataset from Kaggle using kagglehub...")

    # استخدام الكود الخاص بك لتحميل الداتاسيت
    path = kagglehub.dataset_download("abirpaul/amazon-reviews-cell-phones-and-accessories")
    print(f"✅ Download complete. Path to dataset files: {path}")

    # البحث عن ملف JSON داخل المسار الذي تم تحميله
    json_file_path = None
    for f in os.listdir(path):
        if f.endswith('.json'):
            json_file_path = os.path.join(path, f)
            break

    if not json_file_path:
        print("❌ Error: JSON file not found in the downloaded dataset.")
        return

    print(f"⏳ Extracting and splitting data from {json_file_path}...")

    # قراءة أول 10,000 سطر فقط لتسريع العملية وتوفير الذاكرة
    records = []
    with open(json_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 10000:
                break
            try:
                data = json.loads(line)
                if 'reviewText' in data and 'overall' in data:
                    records.append({
                        'reviewText': data['reviewText'],
                        'overall': data['overall']
                    })
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(records)

    # حفظ الملفات كـ CSV لتتناسب مع واجهة النظام
    df.head(100).to_csv("test_100.csv", index=False)
    df.head(1000).to_csv("test_1000.csv", index=False)
    df.head(10000).to_csv("test_10000.csv", index=False)

    print("✅ Files successfully created: test_100.csv | test_1000.csv | test_10000.csv\n")

# 2. اختبار الضغط والسرعة (Load Testing)
API_URL = "http://127.0.0.1:8000/predict" # هنا نحط رابط الموقع
test_files = ["test_100.csv", "test_1000.csv", "test_10000.csv"]

def run_load_test(file_path):
    print(f"🚀 Testing with file: {file_path} ...")

    # التأكد من أن الملف موجود قبل الإرسال
    if not os.path.exists(file_path):
        print(f"❌ File {file_path} not found. Skipping...\n")
        return

    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f, 'text/csv')}
        start_time = time.time()

        try:
            response = requests.post(API_URL, files=files)
            end_time = time.time()

            if response.status_code == 200:
                duration = end_time - start_time
                num_reviews = int(file_path.split('_')[1].split('.')[0])
                throughput = num_reviews / duration

                print(f"✔️ Success! Time taken: {duration:.2f} seconds")
                print(f"📊 Throughput: {throughput:.2f} reviews/second\n")
            else:
                print(f"❌ Failed! API returned status code: {response.status_code}\n")

        except requests.exceptions.ConnectionError:
            print("❌ Connection Error: Ensure the FastAPI server is running on localhost:8000!\n")

# التنفيذ
if __name__ == "__main__":
    # تجهيز الملفات أولاً
    download_and_split_data()

    # تنفيذ الاختبار عليها
    for file in test_files:
        run_load_test(file)