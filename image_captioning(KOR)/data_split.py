#%%
import pandas as pd
import json

with open('C:/Users/PC/Desktop/4학년 2학기/project/coco_data/annotation/MSCOCO_train_val_Korean.json') as f:
    json_data = json.load(f)


json_data=pd.DataFrame(json_data)
json_data['file_path']=json_data['file_path'].apply(lambda x : 'C:/Users/PC/Desktop/4학년 2학기/project/coco_data/annotation/'+x)

def remove_text_len_100(input_list_text):

    list_result = [text for text in input_list_text if len(text)<=100 and text!='']

    return list_result

json_data['caption_ko'] = json_data['caption_ko'].apply(lambda x : remove_text_len_100(x))

json_train_data=json_data[json_data['file_path'].str.contains('train')] 
json_train_data['caption_ko']=json_train_data['caption_ko'].apply(lambda x : '!@#'.join(x))



train_data=json_train_data['caption_ko'].str.split('!@#') 
train_data=train_data.apply(lambda x : pd.Series(x)) 
train_data = train_data.stack().reset_index(level=1, drop=True).to_frame('caption_ko') 
json_train_data.drop('caption_ko',axis=1,inplace=True)

train_data = json_train_data.merge(train_data, left_index=True, right_index=True, how='left') 
train_data.reset_index(drop=True,inplace=True)


print(train_data)

#%%
json_valid_data = json_data[json_data['file_path'].str.contains('val')]
# json_valid_data = json_valid_data.head(8)
json_valid_data['caption_ko'] = json_valid_data['caption_ko'].apply(lambda x: x[0])

valid_data= json_valid_data['caption_ko'].str.split('!@#') 
valid_data= valid_data.apply(lambda x : pd.Series(x)) 
valid_data = valid_data.stack().reset_index(level=1, drop=True).to_frame('caption_ko') 
json_valid_data.drop('caption_ko',axis=1,inplace=True)

valid_data = json_valid_data.merge(valid_data, left_index=True, right_index=True, how='left') 
valid_data.reset_index(drop=True,inplace=True)

print(valid_data)

