#%%
import torch.nn as nn
import torch.optim as optim
from dataloader import model,convert_models_to_fp32,BATCH_SIZE,train_dataloader
from tqdm import tqdm
import torch
import clip


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

EPOCH=5

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
# lr=5e-5   eps=1e-6
# optimizer = optim.Adam(model.parameters(), lr=1e-3,betas=(0.9,0.98),eps=1e-3,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
optimizer = optim.Adam(model.parameters(), lr=1e-2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset


for epoch in range(EPOCH):
    for batch in tqdm(train_dataloader) :
        optimizer.zero_grad()

        images,texts = batch 
    
        images= images.to(device)
        texts = texts.to(device)
    
        x, y = model(images, texts)

        ground_truth = torch.arange(len(images),dtype=torch.long,device=device)

        total_loss = (loss_img(x,ground_truth) + loss_txt(y,ground_truth))/2
        total_loss.backward()
        if device == "cpu":
            optimizer.step()
        else : 
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)

# try:
#     for epoch in tqdm(range(EPOCH)):
#         try:
#             for batch in tqdm(train_dataloader) :
#                 try:
#                     optimizer.zero_grad()

#                     images,texts = batch
#                     images= images.to(device)
#                     texts = texts.to(device)

#                     logits_per_image, logits_per_text = model(images, texts)

#                     ground_truth = torch.arange(len(images),dtype=torch.long,device=device)

#                     total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
#                     total_loss.backward()
#                     if device == "cpu":
#                         optimizer.step()
#                     else : 
#                         convert_models_to_fp32(model)
#                         optimizer.step()
#                         clip.model.convert_weights(model)
#                 except Exception as e:
#                     print('1 : ',e)
#                     pass
#         except Exception as e:
#             print('2 : ',e)
#             pass
                
# except Exception as e:
#     print('3 : ',e)
#     torch.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': total_loss,
#             }, f"C:/Users/PC/Desktop/4학년 2학기/project/coco_data/model_10.pt") #just change to your preferred folder/filename
    
# torch.save({
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': total_loss,
#         }, f"C:/Users/PC/Desktop/4학년 2학기/project/coco_data/model_10.pt") 
# %%
