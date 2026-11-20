"""Reproducible RK1820 W8A8 conversion for raw-head CV models.

Requires x86 RKNN3 Toolkit 1.1.0. Do not install this toolkit inside the
ARM64 inference image. Calibration paths in the dataset must be readable.
"""
import argparse
import hashlib
import json
from pathlib import Path

import onnx
from rknn.api import RKNN


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--onnx',required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--family',required=True,choices=['mobilenetv2','resnet','yolov5','yolov6','yolov8','yolo26','yolo26-segment','yolo26-pose','raft-stereo'])
    a=p.parse_args()
    model=onnx.load(a.onnx)
    weights={t.name for t in model.graph.initializer}
    inputs=[x.name for x in model.graph.input if x.name not in weights]
    config=dict(target_platform='rk1820',core_num=1,quantized_dtype='w8a8',quantized_algorithm='normal',
                quantized_method='channel',input_attrs={name:{'dtype':'uint8','layout':'NHWC'} for name in inputs})
    if a.family in ('mobilenetv2','resnet'):
        config.update(mean_values=[[.485*255,.456*255,.406*255]],std_values=[[.229*255,.224*255,.225*255]])
    elif a.family=='raft-stereo':config.update(mean_values=[[0,0,0]]*2,std_values=[[1,1,1]]*2)
    else:config.update(mean_values=[[0,0,0]],std_values=[[255,255,255]])
    outputs=None
    if a.family.startswith('yolo26'):
        outputs=[]
        for i in range(3):
            outputs += [f'/model.23/one2one_cv2.{i}/one2one_cv2.{i}.2/Conv_output_0',
                        f'/model.23/one2one_cv3.{i}/one2one_cv3.{i}.2/Conv_output_0']
            if a.family=='yolo26-segment':outputs += [f'/model.23/one2one_cv4.{i}/one2one_cv4.{i}.2/Conv_output_0']
            if a.family=='yolo26-pose':outputs += [f'/model.23/one2one_cv4_kpts.{i}/Conv_output_0']
        if a.family=='yolo26-segment':outputs += ['output1']
    elif a.family=='yolov8' and len(model.graph.output)==1:
        outputs=[]
        for i in range(3):outputs += [f'/model.22/cv2.{i}/cv2.{i}.2/Conv_output_0',f'/model.22/cv3.{i}/cv3.{i}.2/Conv_output_0']
    r=RKNN(verbose=False)
    try:
        assert r.config(**config)==0
        kwargs=dict(model=a.onnx)
        if outputs:kwargs['outputs']=outputs
        assert r.load_onnx(**kwargs)==0
        assert r.build(do_quantization=True,dataset=a.dataset)==0
        out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
        assert r.export_rknn(str(out))==0
        out.with_suffix('.conversion.json').write_text(json.dumps(dict(config=config,outputs=outputs,
          onnx_sha256=hashlib.sha256(Path(a.onnx).read_bytes()).hexdigest(),
          calibration_list_sha256=hashlib.sha256(Path(a.dataset).read_bytes()).hexdigest()),indent=2))
    finally:r.release()


if __name__=='__main__':main()
