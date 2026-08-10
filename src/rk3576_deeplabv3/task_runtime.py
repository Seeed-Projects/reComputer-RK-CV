from pathlib import Path
import cv2
import numpy as np
from rknn_runtime import RKNNModel

LABELS = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tv']

def pascal_color(index):
    r = g = b = 0
    value = index
    for shift in range(8):
        r |= ((value >> 0) & 1) << (7 - shift)
        g |= ((value >> 1) & 1) << (7 - shift)
        b |= ((value >> 2) & 1) << (7 - shift)
        value >>= 3
    return b, g, r

class Runtime:
    name = 'deeplabv3'; input_kind = 'image'
    def __init__(self, platform, model_dir):
        self.platform = platform; self.model_dir = Path(model_dir)
        self.model = RKNNModel(self.model_dir/'deeplabv3.rknn', platform)
        self.model_names = ['deeplabv3.rknn']
    def predict(self, image, params):
        original = image.copy(); h, w = image.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(image, (513, 513)), cv2.COLOR_BGR2RGB)
        output = np.asarray(self.model.run(rgb[None])[0])
        if output.ndim != 4: raise ValueError(f'Unexpected output shape: {output.shape}')
        logits = output[0].transpose(1,2,0) if output.shape[1] == len(LABELS) else output[0]
        logits = cv2.resize(logits, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = np.argmax(logits, axis=-1).astype(np.uint8)
        color = np.zeros_like(original)
        present = []
        for index in np.unique(mask):
            color[mask == index] = pascal_color(int(index))
            present.append({'id': int(index), 'class': LABELS[int(index)], 'pixels': int(np.sum(mask == index))})
        preview = cv2.addWeighted(original, 0.5, color, 0.5, 0)
        return {'classes': present, 'width': w, 'height': h}, preview
    def warmup_preview(self, setter):
        image = cv2.imread(str(self.model_dir/'test.jpg')); _, preview = self.predict(image, {}); setter(preview)
def create_runtime(platform, model_dir): return Runtime(platform, model_dir)
