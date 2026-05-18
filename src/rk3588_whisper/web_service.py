import os
import sys
import argparse
import time
import json
import threading
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import shutil

# 导入工具库
from py_utils.whisper_utils import load_audio, log_mel_spectrogram, WhisperTokenizer

# 尝试导入RKNN-Toolkit-Lite2
try:
    from rknnlite.api import RKNNLite
    RKNN_LITE_AVAILABLE = True
except ImportError:
    RKNN_LITE_AVAILABLE = False
    print("Warning: RKNN-Toolkit-Lite2 not available, using fallback")

app = FastAPI(title="RK3588 Whisper Web Service")

# ---------------------------------------------------------
# 模型调度与管理类
# ---------------------------------------------------------
class WhisperModelManager:
    def __init__(self):
        self.encoder = None
        self.decoder = None
        self.tokenizer = None
        self.current_model_size = None
        self.current_language = "en"
        self.lock = threading.Lock()
        self.MAX_TOKENS = 12  # Rockchip exported max tokens window
        
    def load_models(self, model_size, language="en"):
        """
        动态加载/热切换模型
        """
        with self.lock:
            # 如果是相同的模型，不需要重新加载
            if self.current_model_size == model_size:
                if self.current_language != language:
                    self.current_language = language
                    # 语言切换时必须重新实例化 Tokenizer，以重新加载对应的 vocab.txt 并设置 is_zh
                    self.tokenizer = WhisperTokenizer(language=language)
                return True, f"Language updated to {language}"
                
            print(f"Loading Whisper {model_size} models...")
            
            # 1. 释放旧模型
            if self.encoder:
                self.encoder.release()
            if self.decoder:
                self.decoder.release()
                
            # 2. 拼接路径 (以 20s 长度的后缀作为标准)
            base_dir = os.path.dirname(os.path.abspath(__file__))
            encoder_path = os.path.join(base_dir, "model", f"whisper_encoder_{model_size}_20s.rknn")
            decoder_path = os.path.join(base_dir, "model", f"whisper_decoder_{model_size}_20s.rknn")
            
            # 检查文件是否存在
            if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
                return False, f"Model files not found for size '{model_size}' (Expected _20s.rknn)"
                
            # 3. 实际的 RKNN 加载逻辑
            if RKNN_LITE_AVAILABLE:
                # Load Encoder
                self.encoder = RKNNLite()
                ret = self.encoder.load_rknn(encoder_path)
                if ret != 0: return False, "Failed to load encoder model"
                ret = self.encoder.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
                
                # Load Decoder
                self.decoder = RKNNLite()
                ret = self.decoder.load_rknn(decoder_path)
                if ret != 0: return False, "Failed to load decoder model"
                ret = self.decoder.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
            else:
                print("Dummy Load: RKNNLite not available.")
            
            self.tokenizer = WhisperTokenizer(language=language)
            self.current_model_size = model_size
            self.current_language = language
            
            return True, f"Successfully loaded {model_size} model."

    def transcribe(self, audio_array):
        """
        执行推理流程 (Encoder -> Decoder 滑动窗口循环)
        """
        with self.lock:
            if not self.current_model_size:
                raise ValueError("Model not loaded")
                
            # 1. 计算 Log-Mel 频谱图
            mel = log_mel_spectrogram(audio_array)
            
            if not RKNN_LITE_AVAILABLE:
                time.sleep(1) 
                return f"Dummy transcription using {self.current_model_size} model in {self.current_language}."

            # 2. 运行 Encoder 提取特征
            # 输入: [1, 80, 2000]
            outputs = self.encoder.inference(inputs=[mel])
            if outputs is None or outputs[0] is None:
                raise RuntimeError("Encoder inference failed, returned None.")
            audio_features = outputs[0]  # 得到 Encoder 输出特征
            
            # 3. 初始化 Decoder 滑动窗口 Tokens
            initial_tokens = self.tokenizer.get_initial_tokens()
            # 补齐到 MAX_TOKENS 长度，用 0 占位
            # 注意：底层 RKNN 需要 2D 张量，即 [1, MAX_TOKENS]
            tokens_window = np.zeros((1, self.MAX_TOKENS), dtype=np.int64)
            tokens_window[0, :len(initial_tokens)] = initial_tokens
            
            EOT_TOKEN = 50257
            TIMESTAMP_BEGIN = 50364
            
            # 从 tokenizer 获取基础 tokens
            initial_tokens = self.tokenizer.get_initial_tokens()
            
            # === 对齐原版 whisper.py 的解码策略 ===
            # 原版代码使用: tokens = [50258, task_code, 50359, 50363]
            # 然后: tokens = tokens * int(max_tokens/4) => 重复三次占满 12 个位置
            # 这种特殊的重复填充策略极大提升了基于滑动窗口部署的模型的鲁棒性
            
            if self.current_language == "en":
                task_code = 50259
            else:
                task_code = 50260 # ZH
                
            sot_sequence = [50258, task_code, 50359, 50363]
            # 如果 MAX_TOKENS 是 12，那么 12/4 = 3，正好重复 3 次
            repeated_tokens = sot_sequence * (self.MAX_TOKENS // len(sot_sequence))
            
            tokens_window = np.zeros((1, self.MAX_TOKENS), dtype=np.int64)
            tokens_window[0, :len(repeated_tokens)] = repeated_tokens
            
            all_generated_tokens = []
            
            # 4. Decoder 自回归循环 (无 KV Cache，仅滑动 Token 窗口)
            max_decode_steps = 200 # 限制一下最大解码步数，防止死循环
            pop_id = self.MAX_TOKENS
            next_token = 50258 # SOT
            
            for step in range(max_decode_steps):
                # Decoder 需要 2 个输入: [1, MAX_TOKENS] (int64) 和 AUDIO_FEATURES (float)
                dec_outputs = self.decoder.inference(inputs=[tokens_window, audio_features])
                if dec_outputs is None or dec_outputs[0] is None:
                    print(f"Decoder inference failed at step {step}, returned None.")
                    break
                    
                # 输出概率分布，进行 argmax
                logits = dec_outputs[0].flatten()
                
                # 完全对齐 C++ 代码中的 argmax 逻辑
                # start_index = (MAX_TOKENS - 1) * 1 * VOCAB_NUM;
                # max_index = 0; max_value = array[start_index];
                VOCAB_NUM = 51865
                start_idx = (self.MAX_TOKENS - 1) * VOCAB_NUM
                
                # 提取最后一个 Token 对应的概率分布 (长度为 51865)
                token_logits = logits[start_idx : start_idx + VOCAB_NUM]
                next_token = np.argmax(token_logits)
                
                # 如果生成了时间戳则跳过 (与 C++ 逻辑保持一致: if (next_token > timestamp_begin) continue;)
                if next_token > TIMESTAMP_BEGIN:
                    continue
                    
                all_generated_tokens.append(int(next_token))
                
                if next_token == EOT_TOKEN:
                    break
                
                # 滑动窗口策略 (完全对齐原版 whisper.py)
                # 初始 tokens 是重复填充的，随着新词生成，我们需要把前面的 SOT sequence 慢慢挤出去
                if pop_id > 4:
                    pop_id -= 1
                    
                # 将窗口内的元素整体左移 (或者在 Python list 里就是 pop 掉指定位置)
                # 因为我们的 tokens_window 是 numpy array [1, 12]
                # 为了完美对齐: tokens.pop(pop_id); tokens.append(next_token)
                temp_list = tokens_window[0].tolist()
                temp_list.pop(pop_id)
                temp_list.append(int(next_token))
                tokens_window[0] = np.array(temp_list, dtype=np.int64)
                
            # 5. 解码输出
            result_text = self.tokenizer.decode(all_generated_tokens)
            return result_text

model_manager = WhisperModelManager()

# ---------------------------------------------------------
# API 路由
# ---------------------------------------------------------
import uuid
import time
import asyncio
from fastapi import BackgroundTasks

# 任务队列和状态存储
task_store = {}

@app.get("/api/system/status")
async def get_system_status():
    return {
        "status": "success",
        "data": {
            "model_size": model_manager.current_model_size,
            "language": model_manager.current_language,
            "max_tokens": model_manager.MAX_TOKENS,
            "rknn_lite_available": RKNN_LITE_AVAILABLE
        }
    }

@app.post("/api/system/config")
async def update_system_config(model_size: str = Form(...), language: str = Form("en")):
    success, msg = model_manager.load_models(model_size, language)
    if success:
        return {"status": "success", "message": msg}
    else:
        raise HTTPException(status_code=400, detail=msg)

# 为了兼容旧版 Web UI，保留原有的两个接口，内部指向新的系统接口
@app.get("/api/config")
async def get_config_old():
    return await get_system_status()

@app.post("/api/config")
async def update_config_old(model_size: str = Form(...), language: str = Form("en")):
    return await update_system_config(model_size, language)

@app.post("/api/models/whisper/predict")
async def predict_sync(
    file: UploadFile = File(...),
    language: str = Form(None)
):
    """
    同步接口：适合处理 20s 以内的短音频。
    """
    if not model_manager.current_model_size:
        raise HTTPException(status_code=400, detail="Please select and load a model first.")
        
    # 如果请求指定了语言，则临时热切换
    if language and language != model_manager.current_language:
        model_manager.load_models(model_manager.current_model_size, language)
        
    os.makedirs("workspace", exist_ok=True)
    file_path = os.path.join("workspace", file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        t0 = time.time()
        audio = load_audio(file_path)
        duration = len(audio) / 16000.0
        
        # 短音频直接推理
        text = model_manager.transcribe(audio)
        t1 = time.time()
        
        return {
            "status": "success",
            "data": {
                "text": text,
                "language": model_manager.current_language,
                "duration": round(duration, 2),
                "inference_time": round(t1 - t0, 2)
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# 兼容旧版前端上传
@app.post("/api/transcribe")
async def transcribe_file_old(file: UploadFile = File(...)):
    res = await predict_sync(file=file, language=None)
    if res["status"] == "success":
        return {"status": "success", "text": res["data"]["text"]}
    return res

# ---------------------------------------------------------
# 异步长音频任务队列逻辑
# ---------------------------------------------------------
def process_long_audio_task(task_id: str, file_path: str):
    try:
        task_store[task_id]["status"] = "processing"
        
        audio = load_audio(file_path)
        total_samples = len(audio)
        duration = total_samples / 16000.0
        task_store[task_id]["duration"] = round(duration, 2)
        
        # 按照 20s (320000 samples) 进行滑窗切片
        chunk_size = 320000
        chunks = [audio[i:i + chunk_size] for i in range(0, total_samples, chunk_size)]
        
        full_text = ""
        for i, chunk in enumerate(chunks):
            # 简单的静音过滤：如果这段音频能量极低，直接跳过
            if np.max(np.abs(chunk)) < 0.001:
                continue
                
            task_store[task_id]["progress"] = f"Processing chunk {i+1}/{len(chunks)}"
            text = model_manager.transcribe(chunk)
            full_text += text + " "
            
        task_store[task_id]["status"] = "completed"
        task_store[task_id]["result"] = full_text.strip()
        task_store[task_id]["progress"] = "100%"
        
    except Exception as e:
        task_store[task_id]["status"] = "failed"
        task_store[task_id]["message"] = str(e)

@app.post("/api/models/whisper/task")
async def create_async_task(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: str = Form(None)
):
    """
    创建异步分析任务，适合处理大于 20s 的长音频/视频。
    返回 task_id 供轮询。
    """
    if not model_manager.current_model_size:
        raise HTTPException(status_code=400, detail="Please select and load a model first.")
        
    if language and language != model_manager.current_language:
        model_manager.load_models(model_manager.current_model_size, language)
        
    os.makedirs("workspace", exist_ok=True)
    task_id = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename)[1]
    file_path = os.path.join("workspace", f"{task_id}{file_ext}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    task_store[task_id] = {
        "status": "pending",
        "progress": "Queued",
        "result": None,
        "duration": 0,
        "created_at": time.time()
    }
    
    background_tasks.add_task(process_long_audio_task, task_id, file_path)
    
    return {
        "status": "success",
        "data": {
            "task_id": task_id,
            "message": "Task created successfully. Poll /api/models/whisper/task/{task_id} for status."
        }
    }

@app.get("/api/models/whisper/task/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail="Task not found")
        
    return {
        "status": "success",
        "data": task_store[task_id]
    }

# ---------------------------------------------------------
# Web UI (嵌入式 HTML)
# ---------------------------------------------------------
@app.get("/")
async def index():
    html_content = """
    <html>
      <head>
        <title>RK3588 Whisper ASR</title>
        <style>
          body { background-color: #1a1a1a; color: white; font-family: sans-serif; padding: 20px; max-width: 800px; margin: 0 auto;}
          .card { background: #2a2a2a; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
          select, button, input { padding: 10px; border-radius: 5px; border: 1px solid #555; background: #333; color: white; margin: 5px; }
          button { background: #00e676; color: black; font-weight: bold; cursor: pointer; }
          button:hover { background: #00c853; }
          #result { background: #111; padding: 15px; border-radius: 5px; min-height: 100px; font-family: monospace; white-space: pre-wrap; }
        </style>
      </head>
      <body>
        <h1>RK3588 Whisper ASR</h1>
        
        <div class="card">
            <h3>1. Configuration (Hot-Swapping)</h3>
            <label for="modelSize">Model Size: </label>
            <select id="modelSize">
              <option value="base" selected>Base (Recommended)</option>
            </select>
            
            <label for="language" style="margin-left: 20px;">Language: </label>
            <select id="language">
              <option value="en" selected>English</option>
              <option value="zh">Chinese</option>
            </select>
            <button onclick="updateConfig()">Load Model</button>
          <p id="configStatus" style="color: #888;"></p>
        </div>

        <div class="card">
          <h3>2. File Transcription</h3>
          <input type="file" id="audioFile" accept="audio/*,video/*">
          <button onclick="uploadFile()">Transcribe File</button>
          <p id="fileStatus" style="color: #00e676; font-size: 14px;"></p>
        </div>

        <div class="card">
          <h3>3. Real-time Streaming (WebRTC)</h3>
          <button id="recordBtn" onclick="toggleRecording()">Start Recording</button>
          <p id="recordStatus" style="color: #ff3838; font-size: 14px;"></p>
          <p style="color: #888; font-size: 12px;">(Requires HTTPS or localhost to access Microphone)</p>
        </div>

        <div class="card">
          <h3>Transcription Result</h3>
          <div id="result">Waiting for input...</div>
          <button id="downloadBtn" onclick="downloadResult()" style="margin-top: 15px; display: none;">Download Text File</button>
        </div>

        <script>
          // Store latest result for downloading
          let currentTranscription = "";
          let currentFileName = "recording";
          let mediaRecorder;
          let audioChunks = [];

          async function updateConfig() {
            const ms = document.getElementById('modelSize').value;
            const lang = document.getElementById('language').value;
            const fd = new FormData();
            fd.append('model_size', ms);
            fd.append('language', lang);
            
            document.getElementById('configStatus').innerText = "Loading...";
            const res = await fetch('/api/config', { method: 'POST', body: fd });
            const data = await res.json();
            document.getElementById('configStatus').innerText = data.message || data.detail;
          }

          async function uploadFile() {
            const fileInput = document.getElementById('audioFile');
            const file = fileInput.files[0];
            if (!file) return alert('Select a file first!');
            
            currentFileName = file.name.split('.').slice(0, -1).join('.');
            const fd = new FormData();
            fd.append('file', file);
            
            document.getElementById('fileStatus').innerText = "Status: Uploading and analyzing audio...";
            document.getElementById('result').innerText = "Processing file, please wait...";
            document.getElementById('downloadBtn').style.display = 'none';
            
            try {
                const res = await fetch('/api/transcribe', { method: 'POST', body: fd });
                const data = await res.json();
                
                if(data.status === 'success') {
                    document.getElementById('fileStatus').innerText = "Status: Analysis completed successfully!";
                    document.getElementById('result').innerText = data.text;
                    currentTranscription = data.text;
                    document.getElementById('downloadBtn').style.display = 'block';
                } else {
                    document.getElementById('fileStatus').innerText = "Status: Analysis failed!";
                    document.getElementById('result').innerText = "Error: " + data.message;
                }
            } catch(e) {
                document.getElementById('fileStatus').innerText = "Status: Network Error!";
                document.getElementById('result').innerText = e.toString();
            }
          }
          
          async function toggleRecording() {
            const btn = document.getElementById('recordBtn');
            const statusText = document.getElementById('recordStatus');
            
            if(!mediaRecorder || mediaRecorder.state === "inactive") {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];
                    
                    mediaRecorder.ondataavailable = event => {
                        audioChunks.push(event.data);
                    };
                    
                    mediaRecorder.onstop = async () => {
                        statusText.innerText = "Status: Recording stopped. Uploading and analyzing...";
                        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                        const fd = new FormData();
                        fd.append('file', audioBlob, 'web_recording.webm');
                        
                        currentFileName = "web_recording";
                        document.getElementById('result').innerText = "Processing recording, please wait...";
                        document.getElementById('downloadBtn').style.display = 'none';
                        
                        try {
                            const res = await fetch('/api/transcribe', { method: 'POST', body: fd });
                            const data = await res.json();
                            if(data.status === 'success') {
                                statusText.innerText = "Status: Analysis completed successfully!";
                                document.getElementById('result').innerText = data.text;
                                currentTranscription = data.text;
                                document.getElementById('downloadBtn').style.display = 'block';
                            } else {
                                statusText.innerText = "Status: Analysis failed!";
                                document.getElementById('result').innerText = "Error: " + data.message;
                            }
                        } catch(e) {
                            statusText.innerText = "Status: Network Error!";
                        }
                    };
                    
                    mediaRecorder.start();
                    btn.innerText = "Stop Recording";
                    btn.style.background = "#ff3838";
                    statusText.innerText = "Status: Recording in progress...";
                    
                } catch(err) {
                    alert("Microphone access denied or not supported via HTTP. Use localhost or HTTPS.");
                    statusText.innerText = "Status: Failed to access microphone.";
                }
            } else {
                mediaRecorder.stop();
                btn.innerText = "Start Recording";
                btn.style.background = "#00e676";
                // stop all tracks
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
            }
          }

          function downloadResult() {
              if(!currentTranscription) return;
              const dateStr = new Date().toISOString().replace(/:/g, '-').split('.')[0];
              const blob = new Blob([currentTranscription], { type: "text/plain" });
              const url = URL.createObjectURL(blob);
              const a = document.createElement('a');
              a.href = url;
              a.download = `${currentFileName}_${dateStr}.txt`;
              document.body.appendChild(a);
              a.click();
              document.body.removeChild(a);
              URL.revokeObjectURL(url);
          }
        </script>
      </body>
    </html>
    """
    return HTMLResponse(content=html_content)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--default_model', type=str, default='tiny', choices=['tiny', 'base', 'small'])
    parser.add_argument('--default_lang', type=str, default='en')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()

    # 初始化时加载默认模型
    success, msg = model_manager.load_models(args.default_model, args.default_lang)
    print(f"Startup Model Load: {msg}")

    print(f"Starting server at http://{args.host}:{args.port}")
    # uvicorn log_level expects string level like "info", but Python 3.11 logging module is strict about case in some setups.
    # We bypass uvicorn's default logging config by passing log_config=None to prevent ValueError: Unknown level: 'INFO'
    uvicorn.run(app, host=args.host, port=args.port, log_config=None)

if __name__ == '__main__':
    main()
