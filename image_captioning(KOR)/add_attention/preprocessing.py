#%%
import os
import pickle
from collections import Counter

import argparse
import nltk
from PIL import Image
from pycocotools.coco import COCO


class Vocabulary(object):
    
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(json, threshold):
   
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if i % 50000 == 0:
            print("[%d/%d] Tokenized the captions." %(i, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Creates a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Adds the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def resize_image(image):
    width, height = image.size
    if width > height:
        left = (width - height) / 2
        right = width - left
        top = 0
        bottom = height
    else:
        top = (height - width) / 2
        bottom = height - top
        left = 0
        right = width
    image = image.crop((left, top, right, bottom))
    image = image.resize([224, 224], Image.ANTIALIAS)
    return image

dataDir = 'C:/Users/PC/Desktop/COCO/data'
dataType = 'train2017'

caption_path = os.path.join(dataDir, 'annotations/captions_{}.json'.format(dataType))
vocab_path = 'C:/Users/PC/Desktop/COCO/vocab/vocab.pkl'
threshold = 5

vocab = build_vocab(caption_path,threshold)

with open(vocab_path, 'wb') as f:
    pickle.dump(vocab, f)
print("Total vocabulary size: {}".format(len(vocab)))
print("Saved the vocabulary wrapper to '{}'".format(vocab_path))

print("Start resize_image")
splits = ['train', 'val']
for split in splits:
    folder = 'C:/Users/PC/Desktop/COCO/data/{}2017'.format(split)
    resized_folder = 'C:/Users/PC/Desktop/COCO/data/{}2017_resized/'.format(split)
    if not os.path.exists(resized_folder):
        os.makedirs(resized_folder)
    print ('Start resizing {} images.'.format(split))
    image_files = os.listdir(folder)
    num_images = len(image_files)
    for i, image_file in enumerate(image_files):
        with open(os.path.join(folder, image_file), 'r+b') as f:
            with Image.open(f) as image:
                image = resize_image(image)
                image.save(os.path.join(resized_folder, image_file), image.format)
        if i % 5000 == 0:
            print ('Resized images: %d/%d' %(i, num_images))
# %%
