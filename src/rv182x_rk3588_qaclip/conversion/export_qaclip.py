"""Export official QA-CLIP ViT-L/14 with fixed 128-token text inputs.

Word embeddings remain host-side, as in the RKNN3 official example. Both
transformer encoders are converted separately to W8A8 using real inputs.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import torch
from transformers import ChineseCLIPModel, BertTokenizer


class Vision(torch.nn.Module):
    def __init__(self, model):
        super().__init__();self.model=model

    def forward(self,pixel):
        x=self.model.visual_projection(self.model.vision_model(pixel_values=pixel,return_dict=False)[1])
        return x / torch.linalg.vector_norm(x,dim=-1,keepdim=True)


class Text(torch.nn.Module):
    def __init__(self,model):
        super().__init__();self.model=model

    def forward(self,input_embeds,attention_mask,position_ids):
        embeddings=self.model.text_model.embeddings(inputs_embeds=input_embeds,
                    position_ids=position_ids.long(),token_type_ids=torch.zeros_like(position_ids).long())
        mask=(1-attention_mask[:,None,None,:])*-10000.
        hidden=self.model.text_model.encoder(embeddings,attention_mask=mask,return_dict=False)[0]
        x=self.model.text_projection(hidden[:,0,:])
        return x / torch.linalg.vector_norm(x,dim=-1,keepdim=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    src=args.model_dir.resolve();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    model=ChineseCLIPModel.from_pretrained(str(src),torch_dtype=torch.float32,attn_implementation='eager').eval()
    tokenizer=BertTokenizer.from_pretrained(str(src))
    embeddings=model.text_model.embeddings.word_embeddings.weight.detach().numpy()
    np.save(out/'word_embeddings.npy',embeddings)
    tokenizer.save_pretrained(out/'tokenizer')
    prompts=['一辆汽车','一辆卡车','一辆公交车','一辆摩托车','一辆自行车','一个行人','一条公路','一栋房子']
    lines=[]
    for i,prompt in enumerate(prompts):
        tokens=tokenizer(prompt,padding='max_length',max_length=128,truncation=True,return_tensors='np')
        arrays=[embeddings[tokens['input_ids']].astype(np.float32),tokens['attention_mask'].astype(np.float32),np.arange(128,dtype=np.int32)[None]]
        files=[]
        for j,a in enumerate(arrays):
            f=out/f'text-{i}-{j}.npy';np.save(f,a);files.append(str(f))
        lines.append(' '.join(files))
    (out/'text-calibration.txt').write_text('\n'.join(lines)+'\n')
    args=tuple(torch.from_numpy(a) for a in arrays)
    with torch.no_grad():
        torch.onnx.export(Text(model),args,str(out/'qaclip-text.onnx'),opset_version=17,dynamo=False,
             input_names=['input_embeds','attention_mask','position_ids'],output_names=['text_features'])
        torch.onnx.export(Vision(model),(torch.zeros(1,3,224,224),),str(out/'qaclip-vision.onnx'),opset_version=17,dynamo=False,
             input_names=['pixel'],output_names=['image_features'])
    (out/'export.json').write_text(json.dumps(dict(source='TencentARC/QA-CLIP-ViT-L-14',sequence_length=128,
                prompts=prompts,vision_size=224,host_word_embeddings=True,
                logit_scale=float(model.logit_scale.detach().exp())),indent=2))


if __name__=='__main__':main()
