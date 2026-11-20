"""Hardware API regression using identical uploaded images under concurrency."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
import urllib.parse
import uuid
import tempfile

import cv2
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--url',default='http://127.0.0.1:8000')
    p.add_argument('--output',default='validation/api.json')
    a=p.parse_args()
    def request(path,data=None,headers=None):
        try:
            with urllib.request.urlopen(urllib.request.Request(a.url+path,data=data,headers=headers or {}),timeout=90) as r:
                return r.status,r.headers,r.read()
        except urllib.error.HTTPError as e:return e.code,e.headers,e.read()
    status,_,body=request('/api/health');assert status==200
    health=json.loads(body)
    status,_,image=request('/api/snapshot');assert status==200 and image[:2]==b'\xff\xd8'
    boundary=uuid.uuid4().hex
    payload=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="frame.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode()+image+f'\r\n--{boundary}--\r\n'.encode())
    headers={'Content-Type':f'multipart/form-data; boundary={boundary}'}
    def predict(_):
        status,_,body=request('/api/predict',payload,headers);assert status==200,(status,body)
        data=json.loads(body);assert data['success']
        return data['predictions']
    expected=predict(0)
    with ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(predict,range(8)))
    assert all(x==expected for x in results),'Concurrent result mismatch'
    status,_,_=request('/api/config',b'{"obj_thresh":2}',{'Content-Type':'application/json'});assert status==422
    bad=payload.replace(image,b'not an image')
    status,_,_=request('/api/predict',bad,headers);assert status==400
    with urllib.request.urlopen(a.url+'/api/video_feed',timeout=30) as r:
        assert r.status==200 and b'Content-Type: image/jpeg' in r.read(128)
    # Exercise the complete offline pipeline with three actual encoded frames.
    with tempfile.TemporaryDirectory() as directory:
        video=Path(directory)/'smoke-3frames.mp4'
        writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),5,(320,180))
        assert writer.isOpened()
        frame=cv2.resize(cv2.imdecode(np.frombuffer(image,dtype=np.uint8),cv2.IMREAD_COLOR),(320,180))
        for _ in range(3):writer.write(frame)
        writer.release()
        boundary=uuid.uuid4().hex
        payload=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{video.name}"\r\nContent-Type: video/mp4\r\n\r\n'.encode()+video.read_bytes()+f'\r\n--{boundary}--\r\n'.encode())
        status,_,_=request('/api/video/upload',payload,{'Content-Type':f'multipart/form-data; boundary={boundary}'})
        assert status==200
        data=urllib.parse.urlencode({'filename':video.name}).encode()
        status,_,body=request('/api/video/analyze',data,{'Content-Type':'application/x-www-form-urlencoded'})
        assert status==200
        filename=json.loads(body)['output_filename']
        for _ in range(180):
            status,_,body=request('/api/video/status');state=json.loads(body)
            if not state['is_processing']:break
            time.sleep(1)
        assert state['progress']==100 and not state['error'],state
        status,_,data=request('/api/video/download/'+filename);assert status==200
        downloaded=Path(directory)/'result.mp4';downloaded.write_bytes(data)
        cap=cv2.VideoCapture(str(downloaded));decoded=0
        while cap.read()[0]:decoded+=1
        cap.release();assert decoded==3,decoded
    prompt_check=None
    if health['model'].startswith('qaclip'):
        status,_,_=request('/api/prompts',b'{"prompts":[]}',{'Content-Type':'application/json'})
        assert status==422
        payload=json.dumps({'prompts':['一辆汽车','一条公路']}).encode()
        status,_,body=request('/api/prompts',payload,{'Content-Type':'application/json'})
        assert status==200
        prompt_check=json.loads(body)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(health=health,snapshot=True,concurrent_identical_predictions=8,
             invalid_threshold_status=422,invalid_media_status=400,mjpeg=True,predictions=expected,
             prompt_update=prompt_check,offline_video_frames=3),indent=2))
    print('PASS',health['model'],'health/snapshot/MJPEG/8 concurrent requests/invalid inputs')


if __name__=='__main__':main()
