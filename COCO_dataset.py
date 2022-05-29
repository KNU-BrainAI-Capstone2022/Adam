import nltk
import os
import torch
import torch.utils.data as data
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import random
import json

class CoCoDataset(data.Dataset):
    
    def __init__(self, transform, mode, batch_size, vocab_threshold, vocab_file, start_word, 
        end_word, unk_word, annotations_file, vocab_from_file, img_folder):
        self.transform = transform #data augmentation을 진행하기 위한 init
        self.mode = mode # train, validation, test 3가지 있기 때문
        self.batch_size = batch_size
        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word, end_word, unk_word, annotations_file, vocab_from_file)
        self.img_folder = img_folder
        if (self.mode == "train") or (self.mode == "val"): #train or validation일 때
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.anns.keys())
            print("Obtaining caption lengths...")
            all_tokens = [nltk.tokenize.word_tokenize(str(self.coco.anns[self.ids[index]]["caption"]).lower())
                            for index in tqdm(np.arange(len(self.ids)))]
            self.caption_lengths = [len(token) for token in all_tokens]
        
        else: #test일 때
            test_info = json.loads(open(annotations_file).read())
            self.paths = [item["file_name"] for item in test_info["images"]]
        
    def __getitem__(self, index): #image와 caption을 하나씩 꺼내기 위함
       
        if (self.mode == "train") or (self.mode == "val"):
            ann_id = self.ids[index] #annotation 파일에 대한 정보
            caption = self.coco.anns[ann_id]["caption"] #caption 정보
            img_id = self.coco.anns[ann_id]["image_id"] # image 파일에 대한 정보
            path = self.coco.loadImgs(img_id)[0]["file_name"]

            # Convert image to tensor and pre-process using transform
            image = Image.open(os.path.join(self.img_folder, path)).convert("RGB")
            image = self.transform(image)

            #캡션 텐서로 변환
            caption = str(caption).lower() #대문자와 소문자로 인한 차이 발생 없애주기 위해서
            tokens = nltk.tokenize.word_tokenize(caption) #토큰화 시켜주고 전처리가 된 캡션을 만들기 위해서
            caption = []
            caption.append(self.vocab(self.vocab.start_word))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.end_word))
            caption = torch.Tensor(caption).long()

            
            return image, caption #전처리된 이미지와 텐서로 변환시킨 캡션

       
        else: #test일 때
            path = self.paths[index]

            
            PIL_image = Image.open(os.path.join(self.img_folder, path)).convert("RGB") # image에 대해서 텐서로 변환, transform(전처리)
            orig_image = np.array(PIL_image)
            image = self.transform(PIL_image)

            
            return orig_image, image # 원본이미지와 전처리된 이미지의 텐서 return

    def get_indices(self):
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = np.where([self.caption_lengths[i] == sel_length for i in np.arange(len(self.caption_lengths))])[0]
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

    def __len__(self):
        if (self.mode == "train") or (self.mode == "val"):
            return len(self.ids)
        else:
            return len(self.paths)