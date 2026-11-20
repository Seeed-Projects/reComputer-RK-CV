"""Calibrate all DA3 partitions using real images and FP ONNX intermediates."""
from pathlib import Path
import argparse
import cv2
import numpy as np
import onnxruntime as ort


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--images',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=args.models.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    options=ort.SessionOptions();options.intra_op_num_threads=4
    local=ort.InferenceSession(str(source/'da3_base_local.onnx'),options,providers=['CPUExecutionProvider'])
    global_model=ort.InferenceSession(str(source/'da3_base_global.onnx'),options,providers=['CPUExecutionProvider'])
    mean=np.array([.485,.456,.406],dtype=np.float32)
    std=np.array([.229,.224,.225],dtype=np.float32)
    lines_global=[];lines_head=[]
    for i,name in enumerate(args.images.read_text().splitlines()):
        frame=cv2.imread(name)
        image=cv2.cvtColor(cv2.resize(frame,(280,280)),cv2.COLOR_BGR2RGB)
        normalized=((image.astype(np.float32)/255-mean)/std).transpose(2,0,1)[None]
        tokens=local.run(None,{'images':normalized})[0].reshape(1,401,1,768).astype(np.float32)
        filename=out/f'{i:02d}-tokens.npy';np.save(filename,tokens);lines_global.append(str(filename))
        features=global_model.run(None,{'local_tokens':tokens})
        names=[]
        for j,feature in enumerate(features):
            filename=out/f'{i:02d}-feature-{j}.npy'
            np.save(filename,feature.reshape(1,400,1536).astype(np.float32));names.append(str(filename))
        lines_head.append(' '.join(names))
        print('calibration',i,flush=True)
    (out/'global.txt').write_text('\n'.join(lines_global)+'\n')
    (out/'head.txt').write_text('\n'.join(lines_head)+'\n')


if __name__=='__main__':main()
