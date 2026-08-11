from pathlib import Path
import cv2
import numpy as np
from rknn_runtime import RKNNModel

LABELS = ['road','sidewalk','building','wall','fence','pole','traffic light','traffic sign','vegetation','terrain','sky','person','rider','car','truck','bus','train','motorcycle','bicycle']
COLORS = np.array([(128,64,128),(244,35,232),(70,70,70),(102,102,156),(190,153,153),(153,153,153),(250,170,30),(220,220,0),(107,142,35),(152,251,152),(70,130,180),(220,20,60),(255,0,0),(0,0,142),(0,0,70),(0,60,100),(0,80,100),(0,0,230),(119,11,32)], dtype=np.uint8)[:,::-1]
class Runtime:
    name='ppseg'; input_kind='image'
    def __init__(self, platform, model_dir):
        self.platform=platform; self.model_dir=Path(model_dir); self.model=RKNNModel(self.model_dir/'ppseg.rknn',platform); self.model_names=['ppseg.rknn']
    def predict(self,image,params):
        original=image.copy(); h,w=image.shape[:2]
        rgb=cv2.cvtColor(cv2.resize(image,(512,512)),cv2.COLOR_BGR2RGB)
        out=np.asarray(self.model.run(rgb[None])[0])
        if out.ndim != 4: raise ValueError(f'Unexpected output shape: {out.shape}')
        if out.shape[1] == 19: mask=np.argmax(out[0],axis=0).astype(np.uint8)
        elif out.shape[-1] == 19: mask=np.argmax(out[0],axis=-1).astype(np.uint8)
        else: raise ValueError(f'Cannot locate 19 classes in {out.shape}')
        mask=cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST); color=COLORS[mask]
        preview=cv2.addWeighted(original,0.45,color,0.55,0)
        classes=[{'id':int(i),'class':LABELS[int(i)],'pixels':int(np.sum(mask==i))} for i in np.unique(mask)]
        return {'classes':classes,'width':w,'height':h},preview
    def warmup_preview(self,setter):
        image=cv2.imread(str(self.model_dir/'test.png')); _,preview=self.predict(image,{}); setter(preview)
def create_runtime(platform,model_dir): return Runtime(platform,model_dir)
