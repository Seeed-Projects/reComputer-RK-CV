"""Real RKNN3 W8A8 inference using the bundled, previously validated sample."""
import json
from pathlib import Path
import cv2
import numpy as np
from rknn3lite.api import RKNN3Lite

root = Path('/opt/rk182x-runtime/selftest')
runtime = RKNN3Lite()
try:
    assert runtime.load_rknn(str(root / 'mobilenetv2-12.rknn'), str(root / 'mobilenetv2-12.weight')) == 0
    assert runtime.init_runtime(target='rk1820', core_mask=1) == 0
    image = cv2.imread(str(root / 'sample.jpg'))
    assert image is not None
    image = cv2.cvtColor(cv2.resize(image, (224, 224)), cv2.COLOR_BGR2RGB)[None]
    outputs = runtime.inference(inputs=[image], data_format=['nhwc'])
    assert outputs is not None
    scores = np.asarray(outputs[0]).reshape(-1)
    assert scores.size == 1000 and np.isfinite(scores).all()
    top5 = np.argsort(scores)[-5:][::-1].tolist()
    assert 494 in top5, top5
    print(json.dumps(dict(real_npu_inference=True, model='mobilenetv2-12', quantization='w8a8', top5=top5)))
finally:
    runtime.release()
