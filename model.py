import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        #set the hidden size
        self.hidden_size = hidden_size
        
        #embedded layer  # nn.Embedding(num_embeddings, embedding_dim)
        self.embed = nn.Embedding(vocab_size, embed_size)  
        
        #LSTM 
        self.lstm = nn.LSTM(input_size = embed_size,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            batch_first = True,
                            dropout = 0)   #set dropout rate later
        
        # fully-connected layer
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        #initialize the weights for fc layer
        self.init_weights()
        
    
    def forward(self, features, captions):
        #remove end token from captions
        captions = captions[:,:-1]
        
        #embed captions
        captions_embeded = self.embed(captions)
        
        inputs = torch.cat((features.unsqueeze(1), captions_embeded), dim = 1)
        
        #LSTM
        lstm_out, hidden = self.lstm(inputs)
        
        out = self.fc(lstm_out)
        
        return out
        
    
    def init_weights(self):
        #set FC weights as xavier normal
        torch.nn.init.xavier_normal_(self.fc.weight)

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        tokens = []
        count = 0
        while count < max_len:
            out, states = self.lstm(inputs, states)
            out = self.fc(out.squeeze(1))
            predicted = out.argmax(dim = 1)
            tokens.append(predicted.item())
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
            
            count += 1
            if predicted == 1:
                break
                
        return tokens
            
            