from pathlib import Path
import cv2
import numpy as np
from rknn_runtime import RKNNModel
from py_utils.retinaface_official import PriorBox, box_decode, decode_landm, letterbox_resize, nms
class Runtime:
    name='retinaface'; input_kind='image'
    def __init__(self,platform,model_dir):
        self.platform=platform; self.model_dir=Path(model_dir); self.model=RKNNModel(self.model_dir/'retinaface_mobile.rknn',platform); self.model_names=['retinaface_mobile.rknn','retinaface_resnet50.rknn']
    def predict(self,image,params):
        preview=image.copy(); h,w=image.shape[:2]; size=320
        letter,ratio,ox,oy=letterbox_resize(image,(size,size),114); outputs=self.model.run(letter[...,::-1][None]); loc,conf,landmarks=outputs
        priors=PriorBox((size,size)); boxes=box_decode(loc.squeeze(0),priors)*np.array([size]*4); scores=conf.squeeze(0)[:,1]; landmarks=decode_landm(landmarks.squeeze(0),priors)*np.array([size]*10)
        boxes[...,0::2]=np.clip((boxes[...,0::2]-ox)/ratio,0,w); boxes[...,1::2]=np.clip((boxes[...,1::2]-oy)/ratio,0,h); landmarks[...,0::2]=np.clip((landmarks[...,0::2]-ox)/ratio,0,w); landmarks[...,1::2]=np.clip((landmarks[...,1::2]-oy)/ratio,0,h)
        threshold=float(params.get('threshold',0.25)); inds=np.where(scores>0.02)[0]; boxes,landmarks,scores=boxes[inds],landmarks[inds],scores[inds]; order=scores.argsort()[::-1]; boxes,landmarks,scores=boxes[order],landmarks[order],scores[order]; dets=np.hstack((boxes,scores[:,None])).astype(np.float32); keep=nms(dets,0.5)
        result=[]
        for box,lm in zip(dets[keep],landmarks[keep]):
            if float(box[4])<threshold: continue
            coords=[int(v) for v in box[:4]]; points=np.asarray(lm,dtype=int).reshape(5,2); cv2.rectangle(preview,(coords[0],coords[1]),(coords[2],coords[3]),(0,0,255),2)
            for point in points: cv2.circle(preview,tuple(point),2,(0,255,0),-1)
            result.append({'confidence':float(box[4]),'box':coords,'landmarks':points.tolist()})
        return {'faces':result,'count':len(result)},preview
    def warmup_preview(self,setter): image=cv2.imread(str(self.model_dir/'test.jpg')); _,preview=self.predict(image,{}); setter(preview)
def create_runtime(platform,model_dir): return Runtime(platform,model_dir)
