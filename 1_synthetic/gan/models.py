import torch
import torch.nn as nn


class AgentGenerator(nn.Module):
    '''
    class AgentGenerator

    INPUT: noise
    OUTPUT: (x1 and x2)
    LOSS: cost of agent-discriminator and sys-discrminator

    '''
    def __init__(self, opt):
        super(AgentGenerator, self).__init__()

        self.output_size_agent = opt.output_size_agent
        self.latent_dim = opt.latent_dim

        def block(in_feat, out_feat, normalize = True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, 512, normalize=False),
            *block(512, 512, normalize=False),
            *block(512, 512, normalize=False),
            nn.Linear(512, self.output_size_agent),
        )

    def forward(self, z):
        gen_input = z
        agent_o = self.model(gen_input)
        # agent_o = agent_o.view(agent_o.shape[0], self.output_size_agent)
        return agent_o

class AgentDiscriminator(nn.Module):
    '''
    class AgentDiscriminator

    INPUT: output of generator.
    OUTPUT: binary indicator 
    LOSS: Wasserstein distance 
    '''
    def __init__(self, opt):
        super(AgentDiscriminator, self).__init__()

        self.output_size_agent = opt.output_size_agent
        self.latent_dim = opt.latent_dim

        self.model = nn.Sequential(
            nn.Linear(self.output_size_agent, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512,512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,1),
        )

    def forward(self, agent_o):
        disc_input = agent_o
        validity = self.model(disc_input)
        return validity

class SysGenerator(nn.Module):
    """
    class SysGenerator for WGAN-GP

    INPUT: noise
    CONDITION: output of agent generator
    OUTPUT: (revenue)
    LOSS: cost of sys-discriminator
    """
    def __init__(self, opt):
        super(SysGenerator, self).__init__()

        self.output_size = opt.output_size
        self.n_conditions = opt.n_conditions
        self.latent_dim = opt.latent_dim

        # self.label_emb = nn.Embedding(self.n_conditions, self.n_conditions)

        def block(in_feat, out_feat, normalize = True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim + self.n_conditions, 512, normalize=False),
            *block(512, 512, normalize=False),
            *block(512, 512, normalize=False),
            nn.Linear(512, self.output_size),
        )

    def forward(self, z, condition):
        # concatenate the noise and condition
        if type(condition)==list:
            gen_input = torch.cat((*condition, z), -1)
        else:
            gen_input = torch.cat((condition, z), -1)
        sys_o = self.model(gen_input)
        return sys_o

class SysDiscriminator(nn.Module):
    '''
    class SysDiscriminator for WGAN-GP

    INPUT: output of sys-generator.
    CONDITION: output of agent generator
    OUTPUT: binary indicator 
    LOSS: bce
    '''
    def __init__(self, opt):
        super(SysDiscriminator, self).__init__()

        self.output_size = opt.output_size
        self.n_conditions = opt.n_conditions
        self.latent_dim = opt.latent_dim

        # self.label_embedding = nn.Embedding(self.n_conditions, self.n_conditions)

        self.model = nn.Sequential(
            nn.Linear(self.n_conditions+self.output_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512,512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,1),
        )

    def forward(self, sys_o, condition):
        #concatenate the noise and condition
        if type(condition)==list:
            disc_input = torch.cat((sys_o.view(sys_o.size(0), -1), *condition), -1)
        else:
            disc_input = torch.cat((sys_o.view(sys_o.size(0), -1), condition), -1)
        validity = self.model(disc_input)
        return validity

class Sys_critic_mgm(nn.Module):
    '''
    class SysDiscriminator for WGAN-GP

    INPUT: output of sys-generator.
    OUTPUT: a scaler
    LOSS: emd

    '''
    def __init__(self, opt):
        super(Sys_critic_mgm, self).__init__()

        self.output_size = opt.output_size

        self.model = nn.Sequential(
            nn.Linear(self.output_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512,512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,1),
        )

    def forward(self, sys_o):
        validity = self.model((sys_o.view(sys_o.size(0), -1)))
        return validity
