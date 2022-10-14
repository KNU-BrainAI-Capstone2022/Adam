embed_size = 256


encoder = EncoderCNN(embed_size)


if torch.cuda.is_available():
    encoder = encoder.cuda()
    
if torch.cuda.is_available():
    images = images.cuda()

features = encoder(images)

print('type(features):', type(features))
print('features.shape:', features.shape)

hidden_size = 512


vocab_size = len(data_loader.dataset.vocab)


decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

if torch.cuda.is_available():
    decoder = decoder.cuda()
    

if torch.cuda.is_available():
    captions = captions.cuda()

outputs = decoder(features, captions)

print('type(outputs):', type(outputs))
print('outputs.shape:', outputs.shape)