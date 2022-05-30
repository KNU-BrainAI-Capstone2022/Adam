import torch
from torchvision import transforms


transform_train = transforms.Compose([ 
    transforms.Resize(256),                          
    transforms.RandomCrop(224),                      # Image에 대해서 data augmentation 
    transforms.RandomHorizontalFlip(),               
    transforms.ToTensor(),                           
    transforms.Normalize((0.485, 0.456, 0.406),      
                         (0.229, 0.224, 0.225))])


vocab_threshold = 5


batch_size = 10


data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=False)