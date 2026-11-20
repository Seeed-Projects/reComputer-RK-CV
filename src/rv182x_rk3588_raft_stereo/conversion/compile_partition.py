"""W8A8 partition conversion, with no W16/W6 overrides."""
import argparse
import hashlib
import json
from pathlib import Path
from rknn.api import RKNN


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--onnx',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--dataset',required=True)
    p.add_argument('--stage',required=True,choices=['qaclip-text','qaclip-vision','da3-local','da3-global','da3-head'])
    a=p.parse_args()
    config=dict(target_platform='rk1820',core_num=1,quantized_dtype='w8a8',quantized_algorithm='normal',quantized_method='channel')
    if a.stage=='qaclip-text':
        config['input_attrs']={'input_embeds':{'dtype':'float32'},'attention_mask':{'dtype':'float32'},'position_ids':{'dtype':'int32'}}
    elif a.stage=='qaclip-vision':
        config.update(input_attrs={'pixel':{'dtype':'uint8','layout':'NHWC'}},
                mean_values=[[.48145466*255,.4578275*255,.40821073*255]],std_values=[[.26862954*255,.26130258*255,.27577711*255]])
    elif a.stage=='da3-local':
        config.update(input_attrs={'images':{'dtype':'uint8','layout':'NHWC'}},
                mean_values=[[.485*255,.456*255,.406*255]],std_values=[[.229*255,.224*255,.225*255]])
    elif a.stage=='da3-global':config['input_attrs']={'local_tokens':{'dtype':'float32','layout':'NCHW'}}
    else:config['input_attrs']={f'feat_{i}':{'dtype':'float16','layout':'NCHW'} for i in [5,7,9,11]}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    runtime=RKNN(verbose=False)
    try:
        assert runtime.config(**config)==0
        assert runtime.load_onnx(model=a.onnx)==0
        assert runtime.build(do_quantization=True,dataset=a.dataset)==0
        assert runtime.export_rknn(str(out))==0
        out.with_suffix('.json').write_text(json.dumps(dict(stage=a.stage,config=config,do_quantization=True,
                           onnx_sha256=hashlib.sha256(Path(a.onnx).read_bytes()).hexdigest()),indent=2))
    finally:runtime.release()


if __name__=='__main__':main()
