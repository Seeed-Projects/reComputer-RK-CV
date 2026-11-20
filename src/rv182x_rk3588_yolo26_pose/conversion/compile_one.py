import sys,json,time,hashlib
from pathlib import Path
import onnx
from rknn.api import RKNN
def main():
 name=sys.argv[1]; root=Path(__file__).parent; model=root/'models'/f'{name}.onnx'; out=root/'compiled'/name;out.mkdir(parents=True,exist_ok=True)
 m=onnx.load(str(model)); inp=m.graph.input[0].name
 config=dict(target_platform='rk1820',core_num=1,quantized_dtype='w8a8',quantized_algorithm='normal',quantized_method='channel',input_attrs={inp:{'dtype':'uint8','layout':'NHWC'}})
 if name.startswith(('mobilenet','resnet')):
  config.update(mean_values=[[255*.485,255*.456,255*.406]],std_values=[[255*.229,255*.224,255*.225]])
 else: config.update(mean_values=[[0,0,0]],std_values=[[255,255,255]])
 if name.startswith('raftstereo'):
  config.update(mean_values=[[0,0,0],[0,0,0]],std_values=[[1,1,1],[1,1,1]],input_attrs={x.name:{'dtype':'uint8','layout':'NHWC'} for x in m.graph.input if x.name in ['image_left','image_right']})
 outputs=None
 if name.startswith('yolo26'):
  branch='one2one_'; outputs=[]
  for i in range(3):
   outputs += [f'/model.23/{branch}cv2.{i}/{branch}cv2.{i}.2/Conv_output_0',f'/model.23/{branch}cv3.{i}/{branch}cv3.{i}.2/Conv_output_0']
   if '-seg' in name:outputs += [f'/model.23/one2one_cv4.{i}/one2one_cv4.{i}.2/Conv_output_0']
   if '-pose' in name:outputs += [f'/model.23/one2one_cv4_kpts.{i}/Conv_output_0']
  if '-seg' in name:outputs += ['output1']
 if name.startswith('yolov8') and len(m.graph.output)==1:
  outputs=[]
  for i in range(3): outputs += [f'/model.22/cv2.{i}/cv2.{i}.2/Conv_output_0',f'/model.22/cv3.{i}/cv3.{i}.2/Conv_output_0']
 r=RKNN(verbose=False); started=time.time()
 try:
  assert r.config(**config)==0
  kw={'model':str(model)}
  if outputs:kw['outputs']=outputs
  assert r.load_onnx(**kw)==0
  ds=root/('stereo-calibration.txt' if name.startswith('raftstereo') else 'calibration.txt')
  assert r.build(do_quantization=True,dataset=str(ds))==0
  assert r.export_rknn(str(out/f'{name}.rknn'))==0
  (out/'conversion.json').write_text(json.dumps(dict(name=name,config=config,outputs=outputs,source_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),seconds=time.time()-started),indent=2))
 finally:r.release()

if __name__ == '__main__':
 main()
