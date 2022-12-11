# Use Gradio
import numpy as np
import gradio as gr
import torch
from transformers import BartForConditionalGeneration, AutoTokenizer

# Load Pretrained-model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu') # gpu
tokenizer = AutoTokenizer.from_pretrained('gogamza/kobart-base-v2')
model = BartForConditionalGeneration.from_pretrained('D:/jupyter/res/checkpoint-7000').eval()# ckpt-path

def caption2story(caption) :
    # Implement your story generation model here
    input_ids = tokenizer.encode(caption, return_tensors='pt')
    gen_ids = model.generate(input_ids,
                                 do_sample = True,
                                 max_length = 512,
                                 min_length = 64,
                                 repetition_penalty = 1.5,
                                 no_repeat_ngram_size = 3,
                                 temperature = 0.9,
                                 top_k = 50,
                                 top_p = 1.0)
    generated = tokenizer.decode(gen_ids[0])
    return generated

examples = [
    ["노란 셔츠를 입은 소녀가 작은 고양이를 안고 있다."],
    ["사람들은 그들의 오토바이를 타고 어떤 차들 옆을 지나면서 가게와 아파트가 있는 텅 빈 거리를 지난다."]
]

demo = gr.Interface(
    caption2story,
    inputs = gr.inputs.Textbox(lines=5, label="Input Caption"),
    outputs = gr.outputs.Textbox(label = "Generated Text"),
    examples = examples
)
demo.launch()