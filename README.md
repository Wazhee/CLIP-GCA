# Medical Counterfactual Augmentation via Latent Interpolations (Med-CAL)
This is an implementation of Med-CAL for real-time data augmentation using foundation model embeddings 


## How to Implement Med-CLIP
```python
!pip install git+https://github.com/RyanWangZf/MedCLIP.git
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPVisionModel
from medclip import MedCLIPProcessor
import pandas as pd
from PIL import Image
import cv2

#----- Get Embeddings -----
def preprocess_inputs(processor, batch):
    return processor(text=["Chest X-ray Images"],
                      images=batch,
                      return_tensors="pt",
                      padding=True)
    
def get_embeddings(model, dataloader):
    all_embeddings = []
    processor = MedCLIPProcessor()
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    model.from_pretrained()
    model.cuda()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = preprocess_inputs(processor, batch)
            all_embeddings.append(model(**inputs)['img_embeds'].cpu())
    return torch.cat(all_embeddings)

embeddings = get_embeddings(model, dataloader)
```

## How to Implement CXR-CLIP
```python
pip install omegaconf 
pip install albumentations
pip install hydra-core

#----- Get Embeddings -----
def get_embeddings(model, dataloader):
    all_embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            img_emb = model.encode_image(batch["images"])
            all_embeddings.append(img_emb)
    return np.vstack(all_embeddings)

#---- Encode RSNA Dataset ----
dataloader = evaluator.test_dataloader_dict["rsna_pneumonia_test"]
embeddings = get_embeddings(evaluator, dataloader)
```

## Calculate Subgroup Vulnerability
```python
pip install statsmodels
pip install pydicom
pip install statannotations
```

## Cite this work
Kulkarni et al, [*Hidden in Plain Sight*](https://arxiv.org/abs/2402.05713), MIDL 2024.
```
@article{kulkarni2024hidden,
  title={Hidden in Plain Sight: Undetectable Adversarial Bias Attacks on Vulnerable Patient Populations},
  author={Kulkarni, Pranav and Chan, Andrew and Navarathna, Nithya and Chan, Skylar and Yi, Paul H and Parekh, Vishwa S},
  journal={arXiv preprint arXiv:2402.05713},
  year={2024}
}
```
