"""Hardware API regression: image, concurrency, preview, validation, optional video.

Run with Python 3.9+ from the RK3588 host or any machine reaching its API.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--image', type=Path, default=Path(__file__).resolve().parents[1] / 'model/bus.jpg')
    parser.add_argument('--video', type=Path, help='Short video fixture for timestamp and offline-analysis checks')
    parser.add_argument('--output', type=Path, default=Path('verification.json'))
    args = parser.parse_args()

    def request(path, data=None, headers=None):
        req = urllib.request.Request(args.url + path, data=data, headers=headers or {})
        with urllib.request.urlopen(req, timeout=120) as res:
            raw = res.read()
            return json.loads(raw) if 'json' in res.headers.get('Content-Type', '') else raw

    def multipart(path, fields, upload=None):
        boundary = uuid.uuid4().hex
        body = bytearray()
        for k, v in fields.items():
            body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
        if upload:
            name, file = upload
            body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{file.name}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode())
            body.extend(file.read_bytes())
            body.extend(b'\r\n')
        body.extend(f'--{boundary}--\r\n'.encode())
        return request(path, bytes(body), {'Content-Type': f'multipart/form-data; boundary={boundary}'})

    report = {'health': request('/api/health')}
    assert report['health']['accelerator'] == 'rk1820'
    assert b'YOLO26' in request('/')
    predict = lambda: multipart('/api/models/yolo26/predict', {'conf': 0.25, 'iou': 0.45}, ('file', args.image))
    result = predict()
    assert result['success'], result
    assert Counter(p['class'] for p in result['predictions']) == {'person': 4, 'bus': 1}, result
    for p in result['predictions']:
        b = p['box']
        assert 0 <= b['x1'] <= b['x2'] <= result['image']['width']
        assert 0 <= b['y1'] <= b['y2'] <= result['image']['height']
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = list(pool.map(lambda _: predict(), range(8)))
    assert all(r['predictions'] == result['predictions'] for r in concurrent)
    report.update(image=result, concurrent_image_requests=8)
    try:
        request('/api/config', b'{"obj_thresh":2}', {'Content-Type': 'application/json'})
        raise AssertionError('Invalid threshold was accepted')
    except urllib.error.HTTPError as exc:
        assert exc.code == 422
    try:
        multipart('/api/models/yolo26/predict', {'conf': -1}, ('file', args.image))
        raise AssertionError('Invalid request threshold was accepted')
    except urllib.error.HTTPError as exc:
        assert exc.code == 422
    assert request('/api/snapshot').startswith(b'\xff\xd8')
    req = urllib.request.Request(args.url + '/api/video_feed')
    with urllib.request.urlopen(req, timeout=30) as stream:
        assert b'Content-Type: image/jpeg' in stream.read(128)
    report['preview'] = 'JPEG snapshot and MJPEG stream passed'
    if args.video:
        frame = multipart('/api/models/yolo26/predict', {'timestamp': 0.1}, ('video', args.video))
        assert frame['success']
        multipart('/api/video/upload', {}, ('file', args.video))
        started = multipart('/api/video/analyze', {'filename': args.video.name})
        assert started['status'] == 'started'
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            status = request('/api/video/status')
            if not status['is_processing']:
                break
            time.sleep(0.5)
        assert status['progress'] == 100 and not status['error'], status
        output = request('/api/video/download/' + urllib.parse.quote(started['output_filename']))
        assert len(output) > 1000
        report.update(video_frame=frame, offline_analysis=status, output_video_bytes=len(output))
    report['final_health'] = request('/api/health')
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
