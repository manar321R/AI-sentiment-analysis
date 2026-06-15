import os
import json
import pandas as pd
import requests
import time
import kagglehub

# 1. Download and split the dataset using kagglehub
def download_and_split_data():
    print("Downloading dataset from Kaggle using kagglehub...")

    # Download the dataset using kagglehub
    path = kagglehub.dataset_download("abirpaul/amazon-reviews-cell-phones-and-accessories")
    print(f"Download complete. Path to dataset files: {path}")

    # Search for the JSON file inside the downloaded directory
    json_file_path = None
    for f in os.listdir(path):
        if f.endswith('.json'):
            json_file_path = os.path.join(path, f)
            break

    if not json_file_path:
        print("Error: JSON file not found in the downloaded dataset.")
        return

    print(f"Extracting and splitting data from {json_file_path}...")

    # Read only the first 10,000 lines to speed up the process and save memory
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

    # Save the files as CSV to be compatible with the system interface
    df.head(100).to_csv("test_100.csv", index=False)
    df.head(1000).to_csv("test_1000.csv", index=False)
    df.head(10000).to_csv("test_10000.csv", index=False)

    print("Files successfully created: test_100.csv | test_1000.csv | test_10000.csv\n")

# 2. Load and Speed Testing
API_URL = "http://127.0.0.1:8000/predict" # Insert the API endpoint URL here
test_files = ["test_100.csv", "test_1000.csv", "test_10000.csv"]

def run_load_test(file_path):
    print(f"Testing with file: {file_path} ...")

    # Ensure the file exists before sending
    if not os.path.exists(file_path):
        print(f"File {file_path} not found. Skipping...\n")
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

                print(f"Success! Time taken: {duration:.2f} seconds")
                print(f"Throughput: {throughput:.2f} reviews/second\n")
            else:
                print(f"Failed! API returned status code: {response.status_code}\n")

        except requests.exceptions.ConnectionError:
            print("Connection Error: Ensure the FastAPI server is running on localhost:8000!\n")

# Execution block
if __name__ == "__main__":
    # Prepare the files first
    download_and_split_data()

    # Run the load tests on the prepared files
    for file in test_files:
        run_load_test(file)
