import nltk
import pickle
import os.path
from pycocotools.coco import COCO
from collections import Counter

class Vocabulary(object) :
    
    def __init__(self, vocab_threshold, vocab_file='./vocab.pkl',
                start_word="<start>", end_word="<end>", unk_word="<unk>",
                annotations_file = "C:/Users/PC/Desktop/coco/data/annotations/captions_train2017.json",
                vocab_from_file=False) :
        '''
        Initialize the vocabulary.
        
        Args :
        
            < Vocab params >
            vocab_threshold: Minimum word count threshold.
            vocab_file: File containing the vocabulary.
            vocab_from_file: If False, create vocab from scratch & override any existing vocab_file
                             If True, load vocab from from existing vocab_file, if it exists
            < Special words >               
            start_word: Special word denoting sentence start.
            end_word: Special word denoting sentence end.
            unk_word: Special word denoting unknown words.
            
            < File path >
            annotations_file: Path for train annotation file.
        '''
        # vocab params
        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.vocab_from_file = vocab_from_file
        
        # special words
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        
        # file path
        self.annotations_file = annotations_file
        
        # load the vocabulary from file OR build the vocabulary from scratch
        self.get_vocab()
    
    # load the vocabulary from file OR build the vocabulary from scratch
    def get_vocab(self) :
        if os.path.exists(self.vocab_file) & self.vocab_from_file : # vocab file exists : read
            with open(self.vocab_file, 'rb') as f :
                vocab = pickle.load(f)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
            print('Vocabulary successfully loaded from vocab.pkl file !')
        else : # no vocab file : build vocabulary
            self.build_vocab()
            with open(self.vocab_file, 'wb') as f :
                pickle.dump(self, f)
    
    # build vocabulary
    def build_vocab(self) :
        self.init_vocab()
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.add_captions()
    
    # initialize the dictionaries for converting tokens to integers
    def init_vocab(self) :
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
    
    # add token to the vocabulary
    def add_word(self, word) :
        # if it is a "new word", add to the vocabulary
        if not word in self.word2idx :
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1 # vocab word count + 1
    
    # loop over training captions and add all tokens to the vocabulary that meet or exceed the threshold
    def add_captions(self) :
        coco = COCO(self.annotations_file)
        counter = Counter()
        ids = coco.ans.keys()
        
        # 전체 캡션 데이터에 대해 ( 캡션 데이터 각각 대해 토큰화 → 수행 캡션 내 단어 빈도 수 체크 )
        for i, id in enumerate (ids) :
            caption = str(coco.ans[id]['caption'])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)
            
            if i % 100000 == 0 :
                print("[%d/%d] Tokenizing captions..." % (i, len(ids)))
        
        # 전체 캡션 데이터에서 출현 빈도 수가 vocab_threshold이상인 것만 추려낸다
        words = [word for word, cnt in counter.items() if cnt >= self.vocab_threshold]
        
        # 추려낸 단어들을 vocabulary에 추가
        for i, word in enumerate(words):
            self.add_word(word)

    def __call__(self, word) :
        if not word in self.word2idx :
            return self.word2idx[self.unk_word]
        return self.word2idx[word]
    
    # return length of the vocabulary
    def __len__(self) :
        return len(self.word2idx)