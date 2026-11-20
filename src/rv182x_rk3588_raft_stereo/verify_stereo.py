"""Hardware API checks for the paired-input stereo service."""
import base64
import json
from pathlib import Path
import urllib.request
import urllib.error
import uuid


def main():
    base='http://127.0.0.1:8000'
    with urllib.request.urlopen(base+'/api/health') as r:health=json.load(r)
    boundary=uuid.uuid4().hex;parts=[]
    for field,name in [('left','xt_left.jpg'),('right','xt_right.jpg')]:
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; filename="{name}"\r\nContent-Type: image/jpeg\r\n\r\n'.encode()+Path('model',name).read_bytes()+b'\r\n')
    data=b''.join(parts)+f'--{boundary}--\r\n'.encode()
    request=urllib.request.Request(base+'/api/predict',data=data,headers={'Content-Type':f'multipart/form-data; boundary={boundary}'})
    with urllib.request.urlopen(request,timeout=120) as r:result=json.load(r)
    assert result['success']
    pred=result['predictions'][0]
    assert pred['kind']=='disparity' and pred['valid_fraction']>0
    assert base64.b64decode(pred['visualization_png']).startswith(b'\x89PNG')
    with urllib.request.urlopen(base+'/api/sample',timeout=120) as r:
        assert r.headers['Content-Type']=='image/jpeg' and r.read(2)==b'\xff\xd8'
    try:urllib.request.urlopen(urllib.request.Request(base+'/api/predict',data=b''))
    except urllib.error.HTTPError as e:assert e.code==422
    else:raise AssertionError('Missing stereo inputs were accepted')
    Path('validation/api.json').write_text(json.dumps(dict(health=health,paired_upload=True,sample_jpeg=True,missing_input_status=422),indent=2))
    print('PASS stereo health / paired upload / sample JPEG / missing input rejection')


if __name__=='__main__':main()
