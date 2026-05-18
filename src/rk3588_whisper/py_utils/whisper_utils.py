import os
import numpy as np
import librosa
import ffmpeg

# ---------------------------------------------------------
# Whisper Tokenizer 包装类
# ---------------------------------------------------------
import base64

class WhisperTokenizer:
    def __init__(self, language="en"):
        self.language = language
        self.vocab = []
        self.is_zh = (language == "zh")
        
        # 根据语言加载对应的 vocab 文件
        # 使用绝对/相对安全的路径查找模型目录下的资源
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        vocab_path = os.path.join(base_dir, "model", f"vocab_{language}.txt")
        if not os.path.exists(vocab_path):
            print(f"Warning: {vocab_path} not found. Decoder output might be indices only.")
            self.is_dummy = True
        else:
            self.is_dummy = False
            with open(vocab_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        self.vocab.append(parts[1])
                    else:
                        self.vocab.append("")
                        
    def decode(self, tokens):
        if self.is_dummy:
            return " ".join([str(t) for t in tokens])
            
        all_token_str = ""
        for t in tokens:
            if t < len(self.vocab):
                all_token_str += self.vocab[t]
                
        # 按照 C++ 代码中的替换逻辑
        all_token_str = all_token_str.replace("\u0120", " ")
        all_token_str = all_token_str.replace("<|endoftext|>", "")
        all_token_str = all_token_str.replace("\n", "")
        
        if self.is_zh and all_token_str:
            try:
                # C++ 和官方 Python 源码中的特殊 Base64 解码逻辑：
                # 官方源码并没有按空格 split 补齐等号，而是手搓了 Base64 解码！
                # 当它遇到等号 '=' 时，会直接返回一个空格 " "
                # 我们直接引入官方的 base64_decode 逻辑来替换内置库的 base64.b64decode
                
                def get_char_index(c):
                    if 'A' <= c <= 'Z': return ord(c) - ord('A')
                    elif 'a' <= c <= 'z': return ord(c) - ord('a') + (ord('Z') - ord('A') + 1)
                    elif '0' <= c <= '9': return ord(c) - ord('0') + (ord('Z') - ord('A')) + (ord('z') - ord('a')) + 2
                    elif c == '+': return 62
                    elif c == '/': return 63
                    else: return 0

                def custom_base64_decode(encoded_string):
                    if not encoded_string: return ""
                    output_length = len(encoded_string) // 4 * 3
                    decoded_string = bytearray(output_length)
                    index = 0
                    output_index = 0
                    while index < len(encoded_string):
                        if encoded_string[index] == '=':
                            # 这里极其关键：遇到 '=' 意味着遇到了中英文混排或标点符号的断句，官方逻辑是直接用空格替代！
                            return decoded_string[:output_index].decode('utf-8', errors='ignore') + " "
                            
                        first_byte = (get_char_index(encoded_string[index]) << 2) + ((get_char_index(encoded_string[index + 1]) & 0x30) >> 4)
                        decoded_string[output_index] = first_byte

                        if index + 2 < len(encoded_string) and encoded_string[index + 2] != '=':
                            second_byte = ((get_char_index(encoded_string[index + 1]) & 0x0f) << 4) + ((get_char_index(encoded_string[index + 2]) & 0x3c) >> 2)
                            decoded_string[output_index + 1] = second_byte

                            if index + 3 < len(encoded_string) and encoded_string[index + 3] != '=':
                                third_byte = ((get_char_index(encoded_string[index + 2]) & 0x03) << 6) + get_char_index(encoded_string[index + 3])
                                decoded_string[output_index + 2] = third_byte
                                output_index += 3
                            else:
                                output_index += 2
                        else:
                            output_index += 1
                        index += 4
                    return decoded_string[:output_index].decode('utf-8', errors='ignore')

                # 按照空格拆分，分段调用定制的解码逻辑，防止中间的等号导致截断
                parts = all_token_str.split(" ")
                decoded_parts = []
                for p in parts:
                    if p:
                        decoded_parts.append(custom_base64_decode(p))
                all_token_str = "".join(decoded_parts)
                
            except Exception as e:
                print(f"Base64 decode error: {e}")
                
        # 额外清理多余的控制符号 (防止模型输出幻觉或包含未屏蔽的特殊 tokens)
        import re
        all_token_str = re.sub(r'<\|.*?\|>', '', all_token_str)
        all_token_str = all_token_str.strip()
        
        return all_token_str

    def get_initial_tokens(self):
        """
        获取 Decoder 启动时需要的初始 Tokens。
        包含 SOT, Language, Task, NoTimestamps。
        注意：根据 rknn_model_zoo 的 C++ 源码，初始 token 序列通常为4个。
        """
        # 标准的 SOT 序列: [50258, task_code(如50259), 50359, 50363]
        task_code = 50259 if self.language == "en" else 50260
        return [50258, task_code, 50359, 50363]

# ---------------------------------------------------------
# 音频预处理模块
# ---------------------------------------------------------
def load_audio(file_path, sr=16000):
    """
    加载音频文件并重采样到 16kHz
    针对 WebRTC 录制的 webm 等格式，使用 ffmpeg 强制解码为 PCM，避免 librosa 底层 audioread 报错
    """
    try:
        # 使用 ffmpeg 提取音频并重采样，直接输出为 numpy 数组
        out, _ = (
            ffmpeg
            .input(file_path, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True)
        )
        audio = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    except Exception as e:
        print(f"ffmpeg fallback failed: {e}, using librosa directly.")
        # 兜底方案
        audio, _ = librosa.load(file_path, sr=sr)
        
    return audio

def pad_or_trim(array, length=320000):
    """
    不改变总帧数，仅仅做填充或裁剪。
    """
    if len(array) > length:
        return array[:length]
    else:
        return np.pad(array, (0, length - len(array)))

def log_mel_spectrogram(audio, n_mels=80):
    """
    严格对齐 C++ log_mel_spectrogram
    """
    N_FFT = 400
    HOP_LENGTH = 160
    MAX_AUDIO_LENGTH = 320000 # 20s
    
    # 1. 裁剪/填充到 20s
    audio = pad_or_trim(audio, MAX_AUDIO_LENGTH)
    
    # 2. Reflect pad: C++ reflect_pad(audio, padded_audio, N_FFT / 2)
    padded_audio = np.pad(audio, N_FFT // 2, mode='reflect')
    
    # 3. 计算 STFT
    # 注意：librosa.stft 默认自带汉明窗并且做 FFT，和 C++ 的 stft + fftw 对齐
    # 原版 whisper.py 使用了 torch.stft(..., return_complex=True)
    # 我们这里使用 librosa 逼近它，为了更高的识别率，必须抛弃掉最后的一个频段[:-1]
    D = librosa.stft(padded_audio, n_fft=N_FFT, hop_length=HOP_LENGTH, center=False, window='hann')
    magnitudes = np.abs(D[:-1, :])**2
    
    # 4. 计算 Mel 频谱
    # 加载预设的 Mel Filters
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        mel_filters_path = os.path.join(base_dir, "model", "mel_80_filters.txt")
        filters = np.loadtxt(mel_filters_path).reshape(n_mels, N_FFT // 2)
        mel = np.dot(filters, magnitudes)
    except Exception as e:
        print(f"Warning: mel_80_filters.txt not found. Using librosa mel basis as fallback. Error: {e}")
        filters = librosa.filters.mel(sr=16000, n_fft=N_FFT, n_mels=n_mels)
        # librosa 生成的是 201，我们需要对齐截断
        filters = filters[:, :-1] 
        mel = np.dot(filters, magnitudes)
        
    # 5. Clamp & Log10 (Match C++ clamp_and_log_max)
    mel = np.maximum(mel, 1e-10)
    mel = np.log10(mel)
    
    max_val = np.max(mel)
    threshold = max_val - 8.0
    
    mel = np.maximum(mel, threshold)
    mel = (mel + 4.0) / 4.0
    
    # 调整 shape 为模型所需的格式: [1, 80, 2000] 还是 [1, 1, 80, 2000]？
    # C++ 代码中 ENCODER_INPUT_SIZE = CHUNK_LENGTH * 100 = 2000
    # 但是 rknn_inputs_set 要求的 size 是 N_MELS * ENCODER_INPUT_SIZE = 80 * 2000
    # 我们先展平或者调整为标准三维张量
    mel = mel[:, :2000]
    
    # 强制拓展到满足模型输入需求的维度
    # 注意：某些 ONNX 导出的 RKNN 模型可能要求 [1, 1, 80, 2000] 或者 [1, 80, 2000]
    mel = np.expand_dims(mel, axis=0)
    
    # 将 dtype 强制转换为 float32
    return mel.astype(np.float32)
