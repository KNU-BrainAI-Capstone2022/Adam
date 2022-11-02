#%%
import os
import sys
import urllib.request
import json
from data_split import train_data

sample = train_data[train_data['file_path'].str.contains('000000057870')]

kor_sample = sample['caption_ko'][0]
print('AI hub 한국어 caption text :',sample['caption_ko'][0])

eng_sample= sample['captions'][0][0]
print('AI hub 영어 caption text :',eng_sample)


text = eng_sample
inp = 'en'
out = 'ko'

encText = urllib.parse.quote(text)
data = f'source={inp}&target={out}&text=' + encText

url = 'https://openapi.naver.com/v1/papago/n2mt'

client_id = '1RB324JCASWql8p2Aw7h' 
client_secret = 'kB5OawpoKW'

request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id",client_id)
request.add_header("X-Naver-Client-Secret",client_secret)

response = urllib.request.urlopen(request, data=data.encode("utf-8"))
rescode = response.getcode()

if(rescode==200):
    response_body = response.read()
    decode = json.loads(response_body.decode('utf-8'))
    result = decode['message']['result']['translatedText']
    print('AI hub caption 기계 번역 결과 :',result)

else:
    print('Error Code:' + str(rescode))
# %%
