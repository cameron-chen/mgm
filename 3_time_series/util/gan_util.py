from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)
        
        self.__padding = (kernel_size - 1) * dilation
        
    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))

class Agent_Generator(nn.Module):
    '''
    Generator that combines RNN with FC for creating fake time series data (y_1, y_2,...,y_T)
    from the latent variable Z.

    *Note: due to function limitation of pytorch, the framework of generator is fixed
    Args:
        inputs: (torch: tensor) latent variables as inputs to the RNN model has shape
                [batch_size, time_step, sub_sequence_hidden_dims]
    Returns:
        output of generator
    '''
    def __init__(self, time_steps, Dz, Dx, state_size, filter_size, output_activation='sigmoid', bn=False,
                 nlstm=1, nlayer=2, Dy=0, rnn_bn=False, momentum_bn = 1e-2):
        super().__init__()

        self.Dz = Dz
        self.Dy = Dy
        self.Dx = Dx
        self.state_size = state_size
        self.time_steps = time_steps
        self.output_activation = output_activation

        rnn_input_size = Dz+Dy if Dy > 0 else Dz
        self.lstm1 = nn.LSTM(rnn_input_size, state_size, batch_first=True, num_layers=nlstm)
        self.bn1 = nn.BatchNorm1d(filter_size,momentum=momentum_bn)
        self.fc1 = nn.Linear(state_size, filter_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(filter_size, Dx)
        self.tanh = nn.Tanh()

    def forward(self, inputs, y=None):
        assert tuple(inputs.shape[1:]) == (self.time_steps, self.Dz)

        z = inputs
        if y is not None:
            y=y[:,None,:].expand(y.shape[0], self.time_steps, self.Dy)
            z=torch.cat((z, y), -1)
        
        lstm,_ = self.lstm1(z)
        # output shape: (batch size, time steps, state size)

        x = self.fc1(lstm)
        x = self.bn1(x.permute(0,2,1)).permute(0,2,1)
        x = self.relu1(x)
        x = self.fc2(x)
        if self.output_activation=='tanh':
            x = self.tanh(x)
        x = x.view(-1, self.time_steps, self.Dx)

        return x

class Agent_Discriminator(nn.Module):
    '''
    1D CNN Discriminator for H or M
    Args:
        inputs: (numpy array) real time series data (x_1, x_2,...,x_T) and fake samples (y_1, y_2,...,y_T) as inputs
        to the RNN model has shape [batch_size, time_step, x_dims]
    Returns:
        outputs: h or M of shape [batch_size, time_step, J]
    '''

    def __init__(self, time_steps, Dz, Dx, state_size, filter_size, bn=False, kernel_size=5, strides=1,
                 output_activation="tanh", nlayer=2, nlstm=0, momentum_bn = 1e-2):
        super().__init__()

        self.state_size = state_size
        self.time_steps = time_steps
        self.Dz = Dz
        self.Dx = Dx

        # Dense layers
        layers = OrderedDict()
        layers['conv1d1_fc']=CausalConv1d(in_channels=Dx, out_channels=filter_size,
                                          kernel_size=kernel_size, stride=strides)
        if bn:
            layers['bn1_fc']=nn.BatchNorm1d(filter_size, momentum=momentum_bn)
        layers['relu1_fc']=nn.ReLU()
        #* output size: (batch size, filter_size, time_steps)
        for i in range(nlayer-2):
            layers['conv1d{}_fc'.format(i+2)]=CausalConv1d(in_channels=filter_size if i+2==2 else state_size, 
                                                           out_channels=state_size,
                                                           kernel_size=kernel_size, stride=strides)
            if bn:
                layers['bn{}_fc'.format(i+2)]=nn.BatchNorm1d(state_size, momentum=momentum_bn)
            layers['relu{}_fc'.format(i+2)]=nn.ReLU()
        layers['conv1d{}_fc'.format(nlayer)]=CausalConv1d(in_channels=filter_size if nlayer==2 else state_size,
                                                          out_channels=state_size,
                                                          kernel_size=kernel_size, stride=strides)
        if output_activation == 'tanh':
            layers['tanh{}_fc'.format(nlayer)]=nn.Tanh()
        elif output_activation == 'linear':
            pass
        else:
            ValueError('Unknown activation function type')

        self.fc = nn.Sequential(layers)
    
    def forward(self, inputs):
        assert tuple(inputs.shape[1:]) == (self.time_steps, self.Dx)
        
        x = inputs
        x = x.permute(0, 2, 1)
        z = self.fc(x).permute(0, 2, 1)
        #* output shape: (batch size, time steps, state size)

        return z

class Sys_Generator(nn.Module):
    '''
    Generator that combines RNN with FC for creating fake time series data (y_1, y_2,...,y_T)
    from the latent variable Z conditioned on other time series variables.

    *Note: due to function limitation of pytorch, the framework of generator is fixed
    Args:
        (torch: tensor) latent variables: as inputs to the RNN model has shape
            [batch_size, time_step, sub_sequence_hidden_dims]
        condition: time series conditions
    Returns:
        output of generator
    '''
    def __init__(self, time_steps, Dz, Dx, state_size, filter_size, output_activation='sigmoid', bn=False,
                 nlstm=1, nlayer=2, Dy=0, rnn_bn=False, cond_size = 0, momentum_bn = 1e-2):
        super().__init__()

        self.Dz = Dz
        self.Dy = Dy
        self.Dx = Dx
        self.Dcond = cond_size
        self.state_size = state_size
        self.time_steps = time_steps
        self.output_activation = output_activation

        self.lstm1_en = nn.LSTM(cond_size, state_size, batch_first=True)
        self.lstm1_de = nn.LSTM(Dz+Dy+state_size, state_size, batch_first=True)
        self.fc1 = nn.Linear(state_size, filter_size)
        self.bn1 = nn.BatchNorm1d(filter_size,momentum=momentum_bn)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(filter_size, Dx)
        self.tanh = nn.Tanh()

    def forward(self, inputs_z, inputs_cond, inputs_y=None):
        assert tuple(inputs_z.shape[1:]) == (self.time_steps, self.Dz)
        
        z = inputs_z
        if inputs_y is not None:
            y=inputs_y[:,None,:].expand(inputs_y.shape[0], self.time_steps, self.Dy)
            z=torch.cat((z, y), -1) #* z.shape: (batch_size, time_steps, Dz+Dy)
        
        h_en,_ = self.lstm1_en(inputs_cond) #* h_en.shape: (batch_size, timesteps, state_size)
        inp = torch.cat([z, h_en], dim=-1) #* inp.shape: (batch_size, timesteps, Dz+Dy+state_size)

        h_de,_ = self.lstm1_de(inp) #* h_de.shape: (batch_size, timesteps, state_size)

        x = self.fc1(h_de)
        x = self.bn1(x.permute(0,2,1)).permute(0,2,1)
        x = self.relu1(x)
        x = self.fc2(x)
        if self.output_activation=='tanh':
            x = self.tanh(x)
        x = x.view(-1, self.time_steps, self.Dx)

        return x

class Sys_Discriminator(nn.Module):
    '''
    1D CNN Discriminator for H or M
    Args:
        inputs: (numpy array) real time series data (x_1, x_2,...,x_T) and fake samples (y_1, y_2,...,y_T) as inputs
        to the RNN model has shape [batch_size, time_step, x_dims]
    Returns:
        outputs: h or M of shape [batch_size, time_step, J]
    '''

    def __init__(self, time_steps, Dz, Dx, state_size, filter_size, bn=False, kernel_size=5, strides=1,
                 output_activation="tanh", nlayer=2, nlstm=0, cond_size = 0, cond_disc = False, momentum_bn = 1e-2):
        super().__init__()

        self.state_size = state_size
        self.time_steps = time_steps
        self.Dz = Dz
        self.Dx = Dx
        self.Dcond = cond_size
        self.cond_disc = cond_disc

        # Dense layers
        inp_size = Dx+cond_size if cond_disc else Dx
        layers = OrderedDict()
        layers['conv1d1_fc']=CausalConv1d(in_channels=inp_size, out_channels=filter_size,
                                          kernel_size=kernel_size, stride=strides)
        if bn:
            layers['bn1_fc']=nn.BatchNorm1d(filter_size, momentum=momentum_bn)
        layers['relu1_fc']=nn.ReLU()
        #* output size: (batch size, filter_size, time_steps)
        for i in range(nlayer-2):
            layers['conv1d{}_fc'.format(i+2)]=CausalConv1d(in_channels=filter_size if i+2==2 else state_size, 
                                                           out_channels=state_size,
                                                           kernel_size=kernel_size, stride=strides)
            if bn:
                layers['bn{}_fc'.format(i+2)]=nn.BatchNorm1d(state_size, momentum=momentum_bn)
            layers['relu{}_fc'.format(i+2)]=nn.ReLU()
        layers['conv1d{}_fc'.format(nlayer)]=CausalConv1d(in_channels=filter_size if nlayer==2 else state_size,
                                                          out_channels=state_size,
                                                          kernel_size=kernel_size, stride=strides)
        if output_activation == 'tanh':
            layers['tanh{}_fc'.format(nlayer)]=nn.Tanh()
        elif output_activation == 'none':
            pass
        else:
            ValueError('Unknown activation function type')

        self.fc = nn.Sequential(layers)
    
    def forward(self, inputs):
        if self.cond_disc:
            assert tuple(inputs.shape[1:]) == (self.time_steps, self.Dx+self.Dcond)
        else:
            assert tuple(inputs.shape[1:]) == (self.time_steps, self.Dx)
        
        x = inputs
        z = self.fc(x.permute(0, 2, 1)).permute(0, 2, 1)
        #* output shape: (batch size, time steps, state size)

        return z

class Sys_Generator_Mod(nn.Module):
    '''
    Generator that combines RNN with FC for creating fake time series data (y_1, y_2,...,y_T)
    from the latent variable Z conditioned on other time series variables.

    *Note: due to function limitation of pytorch, the framework of generator is fixed
    Args:
        (torch: tensor) latent variables: as inputs to the RNN model has shape
            [batch_size, time_step, sub_sequence_hidden_dims]
        condition: time series conditions
    Returns:
        output of generator
        cond_rec: reconstructed conditions
    '''
    def __init__(self, time_steps, Dz, Dx, state_size, filter_size, output_activation='sigmoid', bn=False,
                 nlstm=1, nlayer=2, Dy=0, rnn_bn=False, cond_size = 0, momentum_bn = 1e-2):
        super().__init__()

        self.Dz = Dz
        self.Dy = Dy
        self.Dx = Dx
        self.Dcond = cond_size
        self.state_size = state_size
        self.time_steps = time_steps
        self.output_activation = output_activation

        self.lstm1_en = nn.LSTM(cond_size, state_size, batch_first=True)
        self.lstm1_de = nn.LSTM(Dz+Dy+state_size, state_size, batch_first=True)
        self.fc1 = nn.Linear(state_size, filter_size)
        self.bn1 = nn.BatchNorm1d(filter_size,momentum=momentum_bn)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(filter_size, Dx)
        self.tanh = nn.Tanh()

        self.fc1_rec = nn.Linear(state_size, filter_size)
        self.bn1_rec = nn.BatchNorm1d(filter_size,momentum=momentum_bn)
        self.relu1_rec = nn.ReLU()
        self.fc2_rec = nn.Linear(filter_size, cond_size)
        self.tanh_rec = nn.Tanh()

    def forward(self, inputs_z, inputs_cond, inputs_y=None):
        assert tuple(inputs_z.shape[1:]) == (self.time_steps, self.Dz)
        
        z = inputs_z
        if inputs_y is not None:
            y=inputs_y[:,None,:].expand(inputs_y.shape[0], self.time_steps, self.Dy)
            z=torch.cat((z, y), -1) #* z.shape: (batch_size, time_steps, Dz+Dy)
        
        h_en,_ = self.lstm1_en(inputs_cond) #* h_en.shape: (batch_size, timesteps, state_size)
        inp = torch.cat([z, h_en], dim=-1) #* inp.shape: (batch_size, timesteps, Dz+Dy+state_size)

        h_de,_ = self.lstm1_de(inp) #* h_de.shape: (batch_size, timesteps, state_size)

        # Generate time series
        x = self.fc1(h_de)
        x = self.bn1(x.permute(0,2,1)).permute(0,2,1)
        x = self.relu1(x)
        x = self.fc2(x)
        if self.output_activation=='tanh':
            x = self.tanh(x)
        x = x.view(-1, self.time_steps, self.Dx)

        # reconstruct conditions
        cond_rec = self.fc1_rec(h_de)
        cond_rec = self.bn1_rec(cond_rec.permute(0,2,1)).permute(0,2,1)
        cond_rec = self.relu1_rec(cond_rec)
        cond_rec = self.fc2_rec(cond_rec)
        if self.output_activation=='tanh':
            cond_rec = self.tanh(cond_rec)
        cond_rec = cond_rec.view(-1, self.time_steps, self.Dcond)

        return x, cond_rec
