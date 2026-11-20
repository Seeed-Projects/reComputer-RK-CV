"""Multi-input / multi-partition adapters for RKNN3 CV examples."""
import base64
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np
from engine import draw as draw_labels, softmax


def load_runtime(path,core_mask):
    from rknn3lite.api import RKNN3Lite
    r=RKNN3Lite()
    try:
        if r.load_rknn(str(path),str(path.with_suffix('.weight'))) != 0:
            raise RuntimeError(f'Failed to load {path}')
        if r.init_runtime(target='rk1820',core_mask=core_mask) != 0:
            raise RuntimeError(f'Failed to initialize {path}')
        return r
    except Exception:r.release();raise


def image_input(frame,w,h,padding=0):
    ratio=min(w/frame.shape[1],h/frame.shape[0])
    rw,rh=round(frame.shape[1]*ratio),round(frame.shape[0]*ratio)
    resized=cv2.resize(frame,(rw,rh))
    result=np.full((h,w,3),padding,dtype=np.uint8);result[:rh,:rw]=resized
    return cv2.cvtColor(result,cv2.COLOR_BGR2RGB)[None],(rw,rh)


def visualization(data,kind):
    data=np.asarray(data,dtype=np.float32).squeeze()
    valid=np.isfinite(data)
    if kind=='depth':valid &= data>0
    if not valid.any():raise RuntimeError(f'No valid {kind} values')
    values=1/np.maximum(data,1e-8) if kind=='depth' else data
    lo,hi=np.percentile(values[valid],[2,98])
    pixels=(np.clip((values-lo)/max(float(hi-lo),1e-8),0,1)*255).astype(np.uint8)
    color=cv2.applyColorMap(pixels,cv2.COLORMAP_TURBO if kind=='depth' else cv2.COLORMAP_PLASMA)
    color[~valid]=0
    ok,png=cv2.imencode('.png',color)
    if not ok:raise RuntimeError('Cannot encode visualization')
    return dict(kind=kind,visualization_png=base64.b64encode(png).decode('ascii'),
                minimum=float(data[valid].min()),maximum=float(data[valid].max()),
                valid_fraction=float(valid.mean()))


class Engine:
    def __init__(self,model_path,weight_path,labels_path,core_mask=1):
        self.metadata=json.loads(Path('project.json').read_text())
        self.family=self.metadata['family'];self.lock=threading.Lock()
        self.core_mask=core_mask
        self.calls=0;self.last_inference_ms=self.last_total_ms=0.
        self.runtimes=[];self.model_path=Path(model_path)
        try:
            if self.family=='depth-anything-v3':
                for s in ['local','global','head']:
                    self.runtimes.append(load_runtime(Path('model')/f'da3_base_{s}.rknn',core_mask))
            elif self.family=='qaclip':
                from transformers import BertTokenizer
                self.tokenizer=BertTokenizer.from_pretrained('model/tokenizer',local_files_only=True)
                self.embedding=np.load('model/word_embeddings.npy',mmap_mode='r')
                self.prompts=[];self.text_features=None
                self.set_prompts(self.metadata['prompts'])
            else:self.runtimes=[load_runtime(self.model_path,core_mask)]
        except Exception:
            for r in self.runtimes:r.release()
            raise

    def set_prompts(self,prompts):
        if not 1<=len(prompts)<=32 or any(not isinstance(s,str) or not s.strip() or len(s)>256 for s in prompts):
            raise ValueError('Provide 1–32 non-empty prompts, each at most 256 characters')
        for prompt in prompts:
            if len(self.tokenizer.encode(prompt,add_special_tokens=True))>128:
                raise ValueError('A prompt exceeds the exported 128-token sequence length')
        with self.lock:
            features=[];t=time.perf_counter()
            # Encode fixed prompts, release text DDR, then retain only vision.
            # Reconfiguration briefly unloads vision to bound card memory use.
            for runtime in self.runtimes:runtime.release()
            self.runtimes=[]
            text=None
            try:
                text=load_runtime(Path('model/qaclip-text.rknn'),self.metadata.get('text_core_mask',1))
                for prompt in prompts:
                    tok=self.tokenizer(prompt,padding='max_length',truncation=True,max_length=128,return_tensors='np')
                    inputs=[np.asarray(self.embedding[tok['input_ids']],dtype=np.float32),
                            tok['attention_mask'].astype(np.float32),np.arange(128,dtype=np.int32)[None]]
                    result=text.inference(inputs=inputs)
                    if result is None:raise RuntimeError('Text inference failed')
                    f=np.asarray(result[0],dtype=np.float32).reshape(-1)
                    features.append(f/max(np.linalg.norm(f),1e-8))
            finally:
                if text is not None:text.release()
            self.runtimes=[load_runtime(Path('model/qaclip-vision.rknn'),self.core_mask)]
            self.text_features=np.stack(features);self.prompts=list(prompts)
            self.text_setup_ms=(time.perf_counter()-t)*1000

    def detect(self,frame,conf=.25,iou=.45,right=None):
        with self.lock:
            start=time.perf_counter()
            if self.family=='raft-stereo':
                if right is None:raise ValueError('RAFT-Stereo requires paired left/right inputs')
                if right.shape != frame.shape:raise ValueError('Stereo input dimensions must match')
                left,(rw,rh)=image_input(frame,240,160);right,_=image_input(right,240,160)
                t=time.perf_counter();out=self.runtimes[0].inference(inputs=[left,right],data_format=['nhwc','nhwc'])
                self.last_inference_ms=(time.perf_counter()-t)*1000
                predictions=[visualization(np.asarray(out[0]).squeeze()[:rh,:rw],'disparity')]
            elif self.family=='depth-anything-v3':
                # Match the official C++ preprocess_image_rgb/convert_image:
                # direct resize to the fixed square input, without letterbox.
                image=cv2.cvtColor(cv2.resize(frame,(280,280)),cv2.COLOR_BGR2RGB)[None]
                t=time.perf_counter()
                local=self.runtimes[0].inference(inputs=[image],data_format=['nhwc'])
                tokens=np.asarray(local[0],dtype=np.float32).reshape(1,401,1,768)
                global_features=self.runtimes[1].inference(inputs=[tokens],data_format=['nchw'])
                features=[np.asarray(f,dtype=np.float16).reshape(1,400,1536) for f in global_features]
                out=self.runtimes[2].inference(inputs=features,data_format=['nchw']*4)
                self.last_inference_ms=(time.perf_counter()-t)*1000
                predictions=[visualization(out[0],'depth')]
                predictions[0]['confidence_mean']=float(np.asarray(out[1]).mean())
            else:
                # CLIP shortest-edge resize followed by center crop.
                h,w=frame.shape[:2];ratio=224/min(h,w)
                image=cv2.resize(frame,(round(w*ratio),round(h*ratio)))
                y,x=(image.shape[0]-224)//2,(image.shape[1]-224)//2
                image=cv2.cvtColor(image[y:y+224,x:x+224],cv2.COLOR_BGR2RGB)[None]
                t=time.perf_counter();out=self.runtimes[0].inference(inputs=[image],data_format=['nhwc'])
                self.last_inference_ms=(time.perf_counter()-t)*1000
                f=np.asarray(out[0],dtype=np.float32).reshape(-1);f/=max(np.linalg.norm(f),1e-8)
                similarity=self.text_features@f
                scores=softmax(similarity*self.metadata.get('logit_scale',100.))
                predictions=[dict(**{'class':self.prompts[int(i)]},class_id=int(i),confidence=float(scores[i]),
                                  cosine_similarity=float(similarity[i])) for i in np.argsort(scores)[::-1]]
            self.calls+=1;self.last_total_ms=(time.perf_counter()-start)*1000
            return predictions,dict(runtime_ms=self.last_inference_ms,total_ms=self.last_total_ms)

    def release(self):
        with self.lock:
            for r in self.runtimes:r.release()


def draw(frame,predictions):
    if predictions and 'visualization_png' in predictions[0]:
        png=base64.b64decode(predictions[0]['visualization_png'])
        heatmap=cv2.imdecode(np.frombuffer(png,dtype=np.uint8),cv2.IMREAD_COLOR)
        return cv2.resize(heatmap,(frame.shape[1],frame.shape[0]))
    labels=[dict(p, **{'class':f"prompt #{p['class_id']}"}) for p in predictions]
    return draw_labels(frame,labels)
