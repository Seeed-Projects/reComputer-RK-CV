"""Run without the web preview competing for the accelerator.

Runtime FPS includes the synchronous Lite call and PCIe transfer. Pipeline FPS
also includes video decoding, preprocessing, task decoding, drawing and JPEG
encoding; excludes HTTP/network and any sleep used to pace a live preview.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import time
from datetime import datetime, timezone

import cv2
import numpy as np
META=json.loads(Path('project.json').read_text())
if META['family'] in ('qaclip','depth-anything-v3'):
    from special_engine import Engine, draw
else:
    from engine import Engine, draw


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',required=True)
    p.add_argument('--video',default='video/test.mp4')
    p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--frames',type=int,default=30)
    p.add_argument('--core-mask',type=lambda x:int(x,0),default=1)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if a.frames < 1 or a.warmup < 0:p.error('Invalid frame count')
    meta=json.loads(Path('project.json').read_text())
    model=Path('model')/f'{a.variant}.rknn'
    weight=model.with_suffix('.weight')
    engine=Engine(model,weight,Path('model')/meta['labels'],a.core_mask)
    cap=cv2.VideoCapture(a.video)
    if not cap.isOpened():raise RuntimeError('Cannot open benchmark input')
    timing=[]; pipelines=[]; result_counts=[]
    try:
        for i in range(a.warmup+a.frames):
            t=time.perf_counter()
            ok,frame=cap.read()
            if not ok:raise RuntimeError('Insufficient test video frames')
            predictions,stats=engine.detect(frame)
            annotated=draw(frame.copy(),predictions)
            h,w=annotated.shape[:2]
            if w>1280:annotated=cv2.resize(annotated,(1280,round(h*1280/w)))
            ok,jpeg=cv2.imencode('.jpg',annotated,[cv2.IMWRITE_JPEG_QUALITY,80])
            if not ok:raise RuntimeError('JPEG encode failed')
            elapsed=time.perf_counter()-t
            if i>=a.warmup:
                timing.append(engine.last_inference_ms/1000)
                pipelines.append(elapsed)
                result_counts.append(len(predictions))
        path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
        path.with_suffix('.jpg').write_bytes(jpeg.tobytes())
        smoke={}
        sample=Path('model/sample.jpg')
        if sample.exists():
            frame_sample=cv2.imread(str(sample))
            predictions_sample,_=engine.detect(frame_sample)
            ids={p['class_id'] for p in predictions_sample if 'class_id' in p}
            expected=set(meta.get('smoke_expected_ids',[]))
            if not expected.issubset(ids):
                raise RuntimeError(f'Official sample class check failed: expected {expected}, got {ids}')
            if meta['family']=='yolo26-segment' and not any(p.get('segments') for p in predictions_sample):
                raise RuntimeError('Official sample has no segmentation polygons')
            if meta['family']=='yolo26-pose' and not any(len(p.get('keypoints',[]))==17 for p in predictions_sample):
                raise RuntimeError('Official sample has no 17-keypoint person')
            cv2.imwrite(str(path.with_suffix('.smoke.jpg')),draw(frame_sample,predictions_sample))
            smoke=dict(expected_class_ids=sorted(expected),actual_class_ids=sorted(ids),predictions=predictions_sample)
        artifacts=meta.get('model_files',[a.variant])
        hashes={name:{ext:hashlib.sha256((Path('model')/(name+ext)).read_bytes()).hexdigest()
                      for ext in ['.rknn','.weight']} for name in artifacts}
        result=dict(variant=a.variant,quantization='w8a8',core_mask=hex(a.core_mask),
                    measured_at_utc=datetime.now(timezone.utc).isoformat(),smoke=smoke,
                    runtime='RKNN3 Lite 1.1.0',host_arch=platform.machine(),
                    input_shape=meta.get('input_shape',[1,224,224,3] if meta['family'] in ('mobilenetv2','resnet') else [1,640,640,3]),
                    warmup=a.warmup,frames=a.frames,preview_width=annotated.shape[1],
                    source_video_shape=list(frame.shape),source_video_sha256=hashlib.sha256(Path(a.video).read_bytes()).hexdigest(),
                    artifact_hashes=hashes,
                    runtime_fps=a.frames/sum(timing),pipeline_fps=a.frames/sum(pipelines),
                    runtime_mean_ms=1000*float(np.mean(timing)),runtime_p95_ms=1000*float(np.percentile(timing,95)),
                    runtime_samples_ms=[1000*x for x in timing],pipeline_samples_ms=[1000*x for x in pipelines],
                    prediction_counts=result_counts,last_predictions=predictions,
                    pipeline='decode + preprocess + synchronous Lite + postprocess + draw + JPEG; no pacing/HTTP')
        if meta['family']=='qaclip':
            result['text_setup_ms']=engine.text_setup_ms
            result['text_prompts']=engine.prompts
            result['text_cache']='Fixed prompts encoded on RK1820 before warmup; FPS measures vision + cached text similarity'
        path.write_text(json.dumps(result,indent=2))
        print(json.dumps({k:v for k,v in result.items() if k.endswith('fps') or k in ('variant','frames')}))
    finally:
        cap.release();engine.release()


if __name__=='__main__':main()
