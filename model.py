import torch
import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module): # Encoder는 CNN
    def __init__(self, embed_size):
        super(Encoder,self).__init__()
        resnet = models.resnet152(pretrained==True)
        
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.init_weights()
    
    
    def forward(self, images):
    
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

    def init_weights(self):
        self.self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)


class Decoder(nn.Module): # Decoder는 RNN
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        
        
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        
        
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        
        self.lstm = nn.LSTM(input_size = embed_size, 
                            hidden_size = hidden_size, 
                            num_layers = num_layers,
                            batch_first = True)
        
        
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
        
    
    def forward(self, features, captions):
    

        captions = captions[:,:-1] 
        embeds = self.word_embeddings(captions)
        
        
        inputs = torch.cat((features.unsqueeze(1), embeds), 1)
        
        lstm_out, hidden = self.lstm(inputs)
        outputs = self.linear(lstm_out)
        
        return outputs
    
    def init_weights(self):
        

        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        
    def sample(self, inputs, states=None, max_len=20):

        predicted_sentence = []
        
        for i in range(max_len):
            
            lstm_out, states = self.lstm(inputs, states)

            lstm_out = lstm_out.squeeze(1)
            lstm_out = lstm_out.squeeze(1)
            outputs = self.linear(lstm_out)
            
            
            target = outputs.max(1)[1]
            
            
            predicted_sentence.append(target.item())
            
            inputs = self.word_embeddings(target).unsqueeze(1)
            
        return predicted_sentence