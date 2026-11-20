import argparse
from ultralytics import YOLO
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--weights',required=True)
    a=p.parse_args()
    YOLO(a.weights).export(format='onnx',opset=19,device='cpu',imgsz=640,dynamic=False,simplify=False,end2end=True)
