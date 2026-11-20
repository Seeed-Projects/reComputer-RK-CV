"""Paired-input stereo web demonstration; a monocular video is never a stereo pair."""
import argparse
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
import uvicorn
from special_engine import Engine, draw

ROOT=Path(__file__).resolve().parent
META=json.loads((ROOT/'project.json').read_text())
options=None;engine=None


@asynccontextmanager
async def lifespan(app):
    global engine
    model=ROOT/'model'/f'{options.variant}.rknn'
    engine=Engine(model,model.with_suffix('.weight'),ROOT/'model/labels.txt',options.core_mask)
    try:yield
    finally:engine.release();engine=None


app=FastAPI(title='reComputer RK-CV · RAFT-Stereo',lifespan=lifespan)


@app.get('/')
def index():return FileResponse(ROOT/'index.html')


@app.get('/api/health')
def health():
    return dict(status='ok' if engine else 'error',model=options.variant,quantization='w8a8',
                runtime='RKNN3 Lite 1.1.0',accelerator='rk1820',host='rk3588',
                inference_count=engine.calls if engine else 0,iterations=1)


def run_pair(left,right):
    if left is None or right is None:raise HTTPException(400,'Invalid paired image inputs')
    if left.shape != right.shape:raise HTTPException(400,'Left/right image sizes must match')
    predictions,timing=engine.detect(left,right=right)
    return dict(success=True,predictions=predictions,timing=timing,image=dict(width=left.shape[1],height=left.shape[0]))


@app.post('/api/models/raft-stereo/predict')
@app.post('/api/predict')
def predict(left:UploadFile=File(...),right:UploadFile=File(...)):
    l=cv2.imdecode(np.frombuffer(left.file.read(),dtype=np.uint8),cv2.IMREAD_COLOR)
    r=cv2.imdecode(np.frombuffer(right.file.read(),dtype=np.uint8),cv2.IMREAD_COLOR)
    return run_pair(l,r)


@app.get('/api/sample')
def sample():
    l=cv2.imread(str(ROOT/'model/xt_left.jpg'));r=cv2.imread(str(ROOT/'model/xt_right.jpg'))
    result=run_pair(l,r)
    image=draw(l,result['predictions'])
    image=cv2.resize(image,(640,round(image.shape[0]*640/image.shape[1])))
    ok,data=cv2.imencode('.jpg',image)
    if not ok:raise HTTPException(500,'Encoding failed')
    return Response(data.tobytes(),media_type='image/jpeg')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--variant',choices=META['variants'],default=os.environ.get('MODEL_VARIANT',META['default_variant']))
    p.add_argument('--core_mask',type=lambda x:int(x,0),default=1)
    p.add_argument('--host',default='0.0.0.0');p.add_argument('--port',type=int,default=8000)
    options=p.parse_args();uvicorn.run(app,host=options.host,port=options.port)
