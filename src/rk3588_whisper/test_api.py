import requests
import time
import os

BASE_URL = "http://127.0.0.1:8000"
TEST_AUDIO = "audio/test_en.wav"

def test_system_status():
    print("\n--- 1. Testing System Status API ---")
    response = requests.get(f"{BASE_URL}/api/system/status")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

def test_sync_transcription():
    print("\n--- 2. Testing Sync Transcription API ---")
    if not os.path.exists(TEST_AUDIO):
        print(f"Error: {TEST_AUDIO} not found.")
        return
        
    with open(TEST_AUDIO, "rb") as f:
        files = {"file": f}
        data = {"language": "en"}
        response = requests.post(f"{BASE_URL}/api/models/whisper/predict", files=files, data=data)
        
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

def test_async_task_queue():
    print("\n--- 3. Testing Async Task Queue API ---")
    if not os.path.exists(TEST_AUDIO):
        print(f"Error: {TEST_AUDIO} not found.")
        return
        
    # 提交任务
    with open(TEST_AUDIO, "rb") as f:
        files = {"file": f}
        data = {"language": "en"}
        response = requests.post(f"{BASE_URL}/api/models/whisper/task", files=files, data=data)
        
    print(f"Submit Task Response: {response.json()}")
    if response.status_code != 200:
        return
        
    task_id = response.json()["data"]["task_id"]
    print(f"Task ID created: {task_id}")
    
    # 轮询状态
    print("Polling task status...")
    while True:
        res = requests.get(f"{BASE_URL}/api/models/whisper/task/{task_id}")
        task_data = res.json()["data"]
        status = task_data["status"]
        progress = task_data["progress"]
        
        print(f"Status: {status} | Progress: {progress}")
        
        if status in ["completed", "failed"]:
            print(f"Final Result: {task_data.get('result', task_data.get('message'))}")
            break
            
        time.sleep(2)

if __name__ == "__main__":
    print("Starting API Tests...")
    test_system_status()
    test_sync_transcription()
    test_async_task_queue()
