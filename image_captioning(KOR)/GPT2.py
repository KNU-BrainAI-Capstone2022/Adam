#%%
from transformers import pipeline, set_seed
from showandtellkor import combination, translation

generator = pipeline('text-generation', model='gpt2')

set_seed(7)

story = generator("a man and a woman are looking at a cell phone,",max_length=300,num_return_sequences=5)[0]['generated_text']
# %% example
print(story)

print(translation(story))

# %% example 3capions

cap = combination()

img2story = generator(cap,max_length=300,num_return_sequences=5)[0]['generated_text']

print(img2story)

print(translation(img2story))



