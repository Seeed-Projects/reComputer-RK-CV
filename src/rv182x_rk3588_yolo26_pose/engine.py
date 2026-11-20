"""Serialized RKNN3 CV inference and task-specific decoding.

Raw YOLO heads follow airockchip/rknn3-model-zoo. All coordinates returned
by the API refer to the original image, including polygons and keypoints.
"""
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -80, 80)))


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def letterbox(frame, size=640):
    h, w = frame.shape[:2]
    ratio = min(size / h, size / w)
    rw, rh = round(w * ratio), round(h * ratio)
    x, y = (size - rw) // 2, (size - rh) // 2
    image = cv2.copyMakeBorder(cv2.resize(frame, (rw, rh)), y, size-rh-y,
                              x, size-rw-x, cv2.BORDER_CONSTANT, value=(114,)*3)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (ratio, x, y, rw, rh)


def nms(boxes, classes, scores, threshold, limit=300):
    keep = []
    for cls in np.unique(classes):
        ids = np.flatnonzero(classes == cls)
        ids = ids[np.argsort(scores[ids])[::-1]]
        while len(ids):
            current, rest = ids[0], ids[1:]
            keep.append(int(current))
            a, b = boxes[current], boxes[rest]
            inter = np.prod(np.maximum(np.minimum(a[2:], b[:, 2:])-
                                       np.maximum(a[:2], b[:, :2]), 0), axis=1)
            union = (np.prod(np.maximum(a[2:]-a[:2], 0)) +
                     np.prod(np.maximum(b[:, 2:]-b[:, :2], 0), axis=1)-inter)
            ids = rest[inter / np.maximum(union, 1e-9) <= threshold]
    return sorted(keep, key=lambda i: float(scores[i]), reverse=True)[:limit]


def decode(outputs, family, conf, iou, size=640):
    """Return boxes, classes, scores, optional per-instance coefficients/keypoints."""
    boxes, scores, classes, extras = [], [], [], []
    stride_heads = 3 if family in ('yolo26-segment', 'yolo26-pose') else len(outputs)//3
    if family == 'yolov5':
        anchors = np.array([[[10,13],[16,30],[33,23]], [[30,61],[62,45],[59,119]],
                            [[116,90],[156,198],[373,326]]], dtype=np.float32)
        for i, output in enumerate(outputs):
            t = np.asarray(output).reshape(3, 85, *output.shape[-2:]).transpose(0,2,3,1)
            h, w = t.shape[1:3]
            gy, gx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            xy = (t[..., :2]*2-.5+np.stack((gx,gy),axis=-1))*(size/w)
            wh = (t[..., 2:4]*2)**2*anchors[i,:,None,None,:]
            b = np.concatenate((xy-wh/2,xy+wh/2),axis=-1).reshape(-1,4)
            probs = (t[...,5:]*t[...,4:5]).reshape(-1,80)
            c, s = probs.argmax(1), probs.max(1)
            mask=s>=conf
            boxes.append(b[mask]); classes.append(c[mask]); scores.append(s[mask])
    else:
        for branch in range(3):
            box, score = [np.asarray(v, dtype=np.float32) for v in
                          outputs[branch*stride_heads:branch*stride_heads+2]]
            h, w = box.shape[-2:]
            if box.shape[1] > 4:
                d = box.reshape(1,4,-1,h,w)
                box = (softmax(d, 2)*np.arange(d.shape[2])[None,None,:,None,None]).sum(2)
            gy, gx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            center = np.stack((gx+.5, gy+.5), axis=0)[None]
            b = np.concatenate((center-box[:,:2],center+box[:,2:]),axis=1)*(size/w)
            logits = score.transpose(0,2,3,1).reshape(-1,score.shape[1])
            # Official YOLOv6 and nine-head YOLOv8 ONNX files include Sigmoid.
            p = logits if family == 'yolov6' or (family == 'yolov8' and len(outputs)==9) else sigmoid(logits)
            c, s = p.argmax(1), p.max(1)
            mask = s >= conf
            boxes.append(b.transpose(0,2,3,1).reshape(-1,4)[mask])
            classes.append(c[mask]); scores.append(s[mask])
            if family in ('yolo26-segment','yolo26-pose'):
                extra = np.asarray(outputs[branch*stride_heads+2], dtype=np.float32)
                e = extra.transpose(0,2,3,1).reshape(-1,extra.shape[1])
                if family == 'yolo26-pose':
                    e=e.reshape(-1,17,3)
                    e[:,:,:2] = (e[:,:,:2]+np.stack((gx+.5,gy+.5),axis=-1).reshape(-1,1,2))*(size/w)
                    e[:,:,2] = sigmoid(e[:,:,2])
                extras.append(e[mask])
    b,c,s = np.concatenate(boxes),np.concatenate(classes),np.concatenate(scores)
    e = np.concatenate(extras) if extras else None
    keep=nms(b,c,s,iou)
    return b[keep],c[keep],s[keep],None if e is None else e[keep]


class Engine:
    def __init__(self, model_path, weight_path, labels_path, core_mask=1):
        from rknn3lite.api import RKNN3Lite
        self.metadata=json.loads(Path(__file__).with_name('project.json').read_text())
        self.family=self.metadata['family']
        self.labels=Path(labels_path).read_text().splitlines()
        expected=1000 if self.family in ('mobilenetv2','resnet') else 1 if self.family=='yolo26-pose' else 80
        if len(self.labels)!=expected:
            raise ValueError(f'{self.family} requires {expected} labels, got {len(self.labels)}')
        self.lock=threading.Lock()
        self.calls=0
        self.last_inference_ms=self.last_total_ms=0.0
        self.runtime=RKNN3Lite()
        try:
            if self.runtime.load_rknn(str(model_path),str(weight_path)) != 0:
                raise RuntimeError('RKNN3 model load failed')
            if self.runtime.init_runtime(target='rk1820',core_mask=core_mask) != 0:
                raise RuntimeError('RKNN3 init failed: check host driver, firmware and proxy')
        except Exception:
            self.runtime.release()
            raise

    def detect(self, frame, conf=.25, iou=.45):
        with self.lock:
            start=time.perf_counter()
            classification=self.family in ('mobilenetv2','resnet')
            if classification:
                # Official RKNN3 demos use direct 224x224 resizing.
                image=cv2.cvtColor(cv2.resize(frame,(224,224)),cv2.COLOR_BGR2RGB)
            else:
                image,transform=letterbox(frame)
            t=time.perf_counter()
            outputs=self.runtime.inference(inputs=[image[None]],data_format=['nhwc'])
            runtime_ms=(time.perf_counter()-t)*1000
            if outputs is None or any(not np.isfinite(o).all() for o in outputs):
                raise RuntimeError('Missing or non-finite model outputs')
            if classification:
                scores=softmax(np.asarray(outputs[0]).reshape(-1).astype(np.float32))
                predictions=[{'class':self.labels[int(i)],'class_id':int(i),
                              'confidence':float(scores[i]),'rank':rank+1}
                             for rank,i in enumerate(np.argsort(scores)[::-1][:5])]
            else:
                b,c,s,e=decode(outputs,self.family,conf,iou)
                ratio,x,y,rw,rh=transform
                original=b.copy()
                b[:,[0,2]]=np.clip((b[:,[0,2]]-x)/ratio,0,frame.shape[1])
                b[:,[1,3]]=np.clip((b[:,[1,3]]-y)/ratio,0,frame.shape[0])
                predictions=[]
                for index,(box,cls,score) in enumerate(zip(b,c,s)):
                    p={'class':self.labels[int(cls)],'class_id':int(cls),'confidence':float(score),
                       'box':dict(zip(('x1','y1','x2','y2'),map(int,box)))}
                    if self.family=='yolo26-pose':
                        kp=e[index].copy()
                        kp[:,0]=np.clip((kp[:,0]-x)/ratio,0,frame.shape[1])
                        kp[:,1]=np.clip((kp[:,1]-y)/ratio,0,frame.shape[0])
                        p['keypoints']=kp.tolist()
                    if self.family=='yolo26-segment':
                        proto=np.asarray(outputs[-1],dtype=np.float32)[0]
                        mask=(e[index]@proto.reshape(proto.shape[0],-1)).reshape(proto.shape[1:])
                        mask=cv2.resize(mask,(640,640))
                        # Crop each instance mask to its model-space detection box.
                        mx1,my1,mx2,my2=np.clip(original[index].astype(int),0,640)
                        crop=np.zeros((640,640),dtype=np.uint8)
                        crop[my1:my2,mx1:mx2]=(mask[my1:my2,mx1:mx2]>0).astype(np.uint8)
                        unpadded=crop[y:y+rh,x:x+rw]
                        contours,_=cv2.findContours(unpadded,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                        p['segments']=[(cv2.approxPolyDP(ct,.8,True).reshape(-1,2)/ratio).round(1).tolist()
                                       for ct in contours if cv2.contourArea(ct)>2]
                    predictions.append(p)
            self.calls+=1
            self.last_inference_ms=runtime_ms
            self.last_total_ms=(time.perf_counter()-start)*1000
            return predictions,dict(runtime_ms=round(runtime_ms,3),total_ms=round(self.last_total_ms,3))

    def release(self):
        with self.lock:
            self.runtime.release()


SKELETON=[(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
          (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]


def draw(frame,predictions):
    scale=max(1.,frame.shape[1]/1280)
    thickness=max(2,round(2*scale))
    for i,p in enumerate(predictions):
        color=(0,230,118)
        if 'box' in p:
            b=p['box']; origin=(b['x1'],max(round(20*scale),b['y1']-round(6*scale)))
            cv2.rectangle(frame,(b['x1'],b['y1']),(b['x2'],b['y2']),color,thickness)
        else:
            origin=(round(15*scale),round((30+i*26)*scale))
        for polygon in p.get('segments',[]):
            pts=np.asarray(polygon,dtype=np.int32).reshape(-1,1,2)
            overlay=frame.copy();cv2.fillPoly(overlay,[pts],color)
            cv2.addWeighted(overlay,.35,frame,.65,0,dst=frame)
        kp=np.asarray(p.get('keypoints',[]))
        if len(kp):
            for a,b in SKELETON:
                if min(kp[a,2],kp[b,2])>.5:
                    cv2.line(frame,tuple(kp[a,:2].astype(int)),tuple(kp[b,:2].astype(int)),color,thickness)
            for x,y,prob in kp:
                if prob>.5:cv2.circle(frame,(int(x),int(y)),thickness+1,(0,180,255),-1)
        cv2.putText(frame,f"{p['class']} {p['confidence']:.2f}",origin,
                    cv2.FONT_HERSHEY_SIMPLEX,.55*scale,color,thickness)
    return frame
