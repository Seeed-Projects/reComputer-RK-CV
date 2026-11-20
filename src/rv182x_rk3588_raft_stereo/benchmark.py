"""Measured paired-image benchmark. One exported recurrent iteration per call."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from special_engine import Engine, draw


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--variant',required=True);p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--frames',type=int,default=30);p.add_argument('--output',required=True)
    p.add_argument('--left',default='model/xt_left.jpg');p.add_argument('--right',default='model/xt_right.jpg')
    a=p.parse_args()
    engine=Engine(Path('model')/f'{a.variant}.rknn',None,None)
    runtime=[];pipeline=[]
    try:
        for i in range(a.warmup+a.frames):
            t=time.perf_counter();left=cv2.imread(a.left);right=cv2.imread(a.right)
            predictions,stats=engine.detect(left,right=right)
            image=draw(left,predictions);h,w=image.shape[:2]
            if w>1280:image=cv2.resize(image,(1280,round(h*1280/w)))
            ok,jpeg=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,80])
            if not ok:raise RuntimeError('JPEG failed')
            elapsed=time.perf_counter()-t
            if i>=a.warmup:runtime.append(engine.last_inference_ms);pipeline.append(elapsed*1000)
        out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
        out.with_suffix('.jpg').write_bytes(jpeg.tobytes())
        result=dict(variant=a.variant,quantization='w8a8',core_mask='0x1',input_shape=[1,160,240,3],
            paired_inputs=2,iterations=1,warmup=a.warmup,frames=a.frames,runtime_fps=a.frames*1000/sum(runtime),
            pipeline_fps=a.frames*1000/sum(pipeline),runtime_p95_ms=float(np.percentile(runtime,95)),
            runtime_samples_ms=runtime,pipeline_samples_ms=pipeline,
            left_sha256=hashlib.sha256(Path(a.left).read_bytes()).hexdigest(),
            right_sha256=hashlib.sha256(Path(a.right).read_bytes()).hexdigest(),
            pipeline='paired JPEG read/decode + preprocessing + synchronous inference + visualization + JPEG',
            last_predictions=[{k:v for k,v in x.items() if k!='visualization_png'} for x in predictions])
        out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
    finally:engine.release()


if __name__=='__main__':main()
