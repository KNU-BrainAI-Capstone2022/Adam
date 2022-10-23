#%%
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from data_split import train_data,valid_data,json_train_data,json_valid_data
import collections
import random
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import clip
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# path = 'C:/Users/PC/Desktop/4학년 2학기/project/coco_data/image/train/train2014/'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model, preprocess = clip.load('ViT-B/32',device=device, jit=False)

class image_title_dataset(Dataset):
    def __init__(self, list_image_path,list_txt):

        self.image_path = list_image_path
        self.title  = clip.tokenize(list_txt) #you can tokenize everything at once in here(slow at the beginning), or tokenize it in the training loop.
        self.list_txt=list_txt ## 이거안하면 밑에 len에서 에러남
    
    def __len__(self):
        return len(self.list_txt) ## self.list_txt=list_txt 안하면에러남
#         return 35700#len(self.list_txt)
    


    def __getitem__(self, idx):
        image = preprocess(Image.open(self.image_path[idx])) # Image from PIL module
        title = self.title[idx]
        return image,title

BATCH_SIZE=1
EPOCH=5

list_image_path=list(train_data['file_path'].values)
list_txt = list(train_data['caption_ko'].values)

dataset = image_title_dataset(list_image_path,list_txt)
train_dataloader = DataLoader(dataset,batch_size = BATCH_SIZE)

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

if device == "cpu":
    model.float()
else :
    clip.model.convert_weights(model) 
