import torch
import torch.nn as nn
import torch.nn.parallel
from miscc.config import cfg
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import models
import torch.utils.model_zoo as model_zoo

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torchvision import models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from miscc.config import cfg


#######################################################################################################
# DO NOT CHANGE THE CLASS NAME, COMPOSITION OF ENCODER CAN BE CHANGED
class RNN_ENCODER(nn.Module):
#######################################################################################################
    def __init__(self, ntoken, ninput=256, drop_prob=0.5, nhidden=128, nlayers=1, bidirectional=True):
        super(RNN_ENCODER, self).__init__()
        self.n_steps = cfg.TEXT.WORDS_NUM
        self.ntoken = ntoken  # size of the dictionary
        self.ninput = ninput  # size of each embedding vector
        self.drop_prob = drop_prob  # probability of an element to be zeroed
        self.nlayers = nlayers  # Number of recurrent layers
        self.bidirectional = bidirectional
        self.rnn_type = cfg.RNN.TYPE
        
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
            
        # number of features in the hidden state
        self.nhidden = nhidden // self.num_directions

        self.define_module()
        self.init_weights()

    def define_module(self):
        '''
        e.g., nn.Embedding, nn.Dropout, nn.LSTM
        '''
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(self.ninput, self.nhidden,
                               self.nlayers, batch_first=True,
                               dropout=self.drop_prob,
                               bidirectional=self.bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.ninput, self.nhidden,
                              self.nlayers, batch_first=True,
                              dropout=self.drop_prob,
                              bidirectional=self.bidirectional)
            
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)    
        

    def forward(self, captions, cap_lens, mask=None):
        '''
        1. caption -> embedding
        2. pack_padded_sequence (embedding, cap_lens)
        3. for rnn, hidden is used
        4. sentence embedding should be returned
        '''
        
        # input: torch.LongTensor of size batch x n_steps
        # --> emb: batch x n_steps x ninput
        emb = self.drop(self.encoder(captions))
        #
        # Returns: a PackedSequence object
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True)
        # #hidden and memory (num_layers * num_directions, batch, hidden_size):
        # tensor containing the initial hidden state for each element in batch.
        # #output (batch, seq_len, hidden_size * num_directions)
        # #or a PackedSequence object:
        # tensor containing output features (h_t) from the last layer of RNN\
        
        output, hidden = self.rnn(emb)
        # PackedSequence object
        # --> (batch, seq_len, hidden_size * num_directions)
        output = pad_packed_sequence(output, batch_first=True)[0]
        # output = self.drop(output)
        # --> batch x hidden_size*num_directions x seq_len
        # words_emb = output.transpose(1, 2)
        # --> batch x num_directions*hidden_size
        if self.rnn_type == 'LSTM':
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)
        
        return sent_emb

#######################################################################################################
# DO NOT CHANGE 
class CNN_ENCODER(nn.Module):
#######################################################################################################
    def __init__(self, nef):
        super(CNN_ENCODER, self).__init__()
        '''
        nef: size of image feature
        '''

        '''
        any pretrained cnn encoder can be loaded if necessary
        '''

        self.define_module()
    def define_module(self):
        '''
        '''
        self.model = models.resnet101(pretrained = True)
        for params in self.model.parameters():
            params.required_grad = False
            
        self.model.fc = nn.Linear(2048, cfg.TEXT.EMBEDDING_DIM)
        self.model.fc.required_grad = True
        
        
    def forward(self, x):
        '''
        '''
        x = self.model(x)
        return x

#######################################################################################################
# DO NOT CHANGE  
class GENERATOR(nn.Module):
#######################################################################################################
    def __init__(self):
        super(GENERATOR, self).__init__()
        '''
        '''
        self.conv1 = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( cfg.GAN.Z_DIM + cfg.TEXT.EMBEDDING_DIM, cfg.GAN.GF_DIM * 16, 4, 1, 0, bias=True),
            nn.BatchNorm2d(cfg.GAN.GF_DIM * 16),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            # state size. (cfg.GAN.GF_DIM*16) x 4 x 4
            nn.ConvTranspose2d( cfg.GAN.GF_DIM * 16, cfg.GAN.GF_DIM * 8, 4, 2, 1, bias=True),
            nn.BatchNorm2d(cfg.GAN.GF_DIM * 8),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            # state size. (cfg.GAN.GF_DIM*8) x 8 x 8
            nn.ConvTranspose2d(cfg.GAN.GF_DIM * 8, cfg.GAN.GF_DIM * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(cfg.GAN.GF_DIM * 4),
            nn.ReLU(True)
        )
        self.conv4 = nn.Sequential(
            # state size. (cfg.GAN.GF_DIM*4) x 16 x 16
            nn.ConvTranspose2d( cfg.GAN.GF_DIM * 4, cfg.GAN.GF_DIM * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(cfg.GAN.GF_DIM * 2),
            nn.ReLU(True)
        )
        self.conv5 = nn.Sequential(
            # state size. (cfg.GAN.GF_DIM*2) x 32 x 32
            nn.ConvTranspose2d( cfg.GAN.GF_DIM * 2, cfg.GAN.GF_DIM, 4, 2, 1, bias=True),
            nn.BatchNorm2d(cfg.GAN.GF_DIM),
            nn.ReLU(True)
        )
        self.conv6 = nn.Sequential(
            # state size. (cfg.GAN.GF_DIM) x 64 x 64
            nn.ConvTranspose2d( cfg.GAN.GF_DIM, 3, 4, 2, 1, bias=True),
            nn.Tanh()
            # state size. 3 x 128 x 128
        )
        
        self.weights_init()
        
    # custom weights initialization
    def weights_init(self):
        classname = self.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(self.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(self.weight.data, 1.0, 0.02)
            nn.init.constant_(self.bias.data, 0)
            
        
    def forward(self, z_code, sent_emb):
        """
        z_code: batch x cfg.GAN.Z_DIM (100)
        sent_emb: batch x cfg.TEXT.EMBEDDING_DIM (256)
        return: generated image
        """
        # print(z_code.shape)
        # print(sent_emb.shape)
        x = torch.cat((z_code, sent_emb.reshape(-1,cfg.TEXT.EMBEDDING_DIM,1,1)), dim=1)
        # print(x.shape)

        x = self.conv1(x)
        # state size. (cfg.GAN.GF_DIM*16) x 4 x 4
        # print(x.shape)
        
        x = self.conv2(x)
        # state size. (cfg.GAN.GF_DIM*8) x 8 x 8
        # print(x.shape)
        
        x = self.conv3(x)
        # state size. (cfg.GAN.GF_DIM*4) x 16 x 16
        # print(x.shape)
        
        x = self.conv4(x)
        # state size. (cfg.GAN.GF_DIM*2) x 32 x 32
        # print(x.shape)
        
        x = self.conv5(x)
        # state size. (cfg.GAN.GF_DIM) x 64 x 64
        # print(x.shape)
        fake_imgs = self.conv6(x)
        
        # state size. 3 x 128 x 128
        # print(x.shape)

        return fake_imgs



#######################################################################################################
# DO NOT CHANGE 
class DISCRIMINATOR(nn.Module):
#######################################################################################################
    def __init__(self, b_jcu=True):
        super(DISCRIMINATOR, self).__init__()
        '''
        '''
        self.conv1 = nn.Sequential(
            # input is (3) x 128 x 128
            nn.Conv2d(3, cfg.GAN.DF_DIM, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            # state size. (cfg.GAN.DF_DIM) x 64 x 64
            nn.Conv2d(cfg.GAN.DF_DIM, cfg.GAN.DF_DIM * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(cfg.GAN.DF_DIM * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            # state size. (cfg.GAN.DF_DIM*2) x 32 x 32
            nn.Conv2d(cfg.GAN.DF_DIM * 2, cfg.GAN.DF_DIM * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(cfg.GAN.DF_DIM * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            # state size. (cfg.GAN.DF_DIM*4) x 16 x 16
            nn.Conv2d(cfg.GAN.DF_DIM * 4, cfg.GAN.DF_DIM * 8, 4, 2, 1, bias=True),
            nn.BatchNorm2d(cfg.GAN.DF_DIM * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv5 = nn.Sequential(
            # state size. (cfg.GAN.DF_DIM*8) x 8 x 8
            nn.Conv2d(cfg.GAN.DF_DIM * 8, cfg.GAN.DF_DIM * 16, 4, 2, 1, bias=True),
            nn.BatchNorm2d(cfg.GAN.DF_DIM * 16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv6 = nn.Sequential(
            # state size. (cfg.GAN.DF_DIM*16) x 4 x 4
            nn.Conv2d(cfg.GAN.DF_DIM * 16, 1, 4, 1, 0, bias=True),
            nn.Sigmoid()
        )
        
        self.weights_init()
    # custom weights initialization
    def weights_init(self):
        classname = self.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(self.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(self.weight.data, 1.0, 0.02)
            nn.init.constant_(self.bias.data, 0)
            
    def forward(self, x_var):
        '''
        '''
        # input is (3) x 128 x 128
        x_var = self.conv1(x_var)
        
        # state size. (cfg.GAN.DF_DIM) x 64 x 64
        x_var = self.conv2(x_var)
        
        # state size. (cfg.GAN.DF_DIM*2) x 32 x 32
        x_var = self.conv3(x_var)
        
        # state size. (cfg.GAN.DF_DIM*4) x 16 x 16
        x_var = self.conv4(x_var)
        
        # state size. (cfg.GAN.DF_DIM*8) x 8 x 8
        x_var = self.conv5(x_var)
        
        # state size. (cfg.GAN.DF_DIM*16) x 4 x 4
        x_var = self.conv6(x_var)
        
        return x_var
