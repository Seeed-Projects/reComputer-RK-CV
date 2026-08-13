from pathlib import Path
import cv2
import numpy as np
from rknn_runtime import RKNNModel

SIZE=448
class Runtime:
    name='mobilesam'; input_kind='image'
    def __init__(self,platform,model_dir):
        self.platform=platform; self.model_dir=Path(model_dir); self.encoder=RKNNModel(self.model_dir/'mobilesam_encoder.rknn',platform,0); self.decoder=RKNNModel(self.model_dir/'mobilesam_decoder.rknn',platform,1); self.model_names=['mobilesam_encoder.rknn','mobilesam_decoder.rknn']
    @staticmethod
    def resized_shape(h,w):
        scale=SIZE/max(h,w); return int(h*scale+0.5),int(w*scale+0.5)
    def predict(self,image,params):
        original=image.copy(); h,w=image.shape[:2]; nh,nw=self.resized_shape(h,w)
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB); rgb=cv2.resize(rgb,(nw,nh)); rgb=cv2.copyMakeBorder(rgb,0,SIZE-nh,0,SIZE-nw,cv2.BORDER_CONSTANT,value=0).astype(np.float32)[None]
        embedding=np.asarray(self.encoder.run(rgb)[0])
        if embedding.shape == (1,256,28,28):
            embedding=np.ascontiguousarray(embedding.transpose(0,2,3,1))
        elif embedding.shape != (1,28,28,256):
            raise RuntimeError(f'Unexpected MobileSAM embedding shape: {embedding.shape}')
        prompt_coords=params.get('point_coords') or [[0,0],[w-1,h-1]]
        prompt_labels=params.get('point_labels') or [2,3]
        raw_coords=np.asarray(prompt_coords,dtype=np.float32).reshape(-1,2)
        raw_coords[:,0]=np.clip(raw_coords[:,0],0,w-1); raw_coords[:,1]=np.clip(raw_coords[:,1],0,h-1)
        labels=np.asarray(prompt_labels,dtype=np.float32).reshape(1,-1)
        coords=raw_coords.reshape(1,-1,2).copy()
        if coords.shape[1] != 2 or labels.shape[1] != 2: raise ValueError('This converted decoder requires exactly two prompt points')
        coords[...,0]*=nw/w; coords[...,1]*=nh/h
        mask_input=np.zeros((1,112,112,1),dtype=np.float32); has_mask=np.zeros(1,dtype=np.float32)
        scores,masks=self.decoder.run([embedding,coords,labels,mask_input,has_mask])[:2]
        best=int(np.argmax(np.asarray(scores).reshape(-1))); low=np.asarray(masks)[0,best]
        resized=cv2.resize(low,(SIZE,SIZE),interpolation=cv2.INTER_LINEAR)[:nh,:nw]
        mask=cv2.resize(resized,(w,h),interpolation=cv2.INTER_LINEAR)>0
        overlay=original.copy(); color=np.zeros_like(original); color[:]=(30,144,144); overlay[mask]=cv2.addWeighted(original[mask],0.5,color[mask],0.5,0)
        draw_coords=raw_coords.astype(int); draw_labels=labels.reshape(-1).astype(int).tolist()
        if draw_labels == [2,3]:
            cv2.rectangle(overlay,tuple(draw_coords[0]),tuple(draw_coords[1]),(0,255,0),2)
        else:
            for point,label in zip(draw_coords,draw_labels):
                if label in (0,1): cv2.circle(overlay,tuple(point),7,(0,255,0) if label == 1 else (0,0,255),-1)
        return {'iou_scores':np.asarray(scores).reshape(-1).astype(float).tolist(),'selected_mask':best,'mask_pixels':int(mask.sum()),'prompt':{'point_coords':draw_coords.tolist(),'point_labels':draw_labels}},overlay
    def warmup_preview(self,setter):
        image=cv2.imread(str(self.model_dir/'picture.jpg')); _,preview=self.predict(image,{}); setter(preview)
def create_runtime(platform,model_dir): return Runtime(platform,model_dir)
