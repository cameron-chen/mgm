################
# import the packages
################

import os
import pickle
from copy import deepcopy

import optuna
import torch
from gan import help, losses, models
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

################
# main
################

def main(opt,timestamp=None, trial=None):
    # --------------- input ---------------
    lambda_gp = opt.lambda_gp # hyperparameter lambda
    long_dtype, float_dtype = help.get_dtypes(opt)

    cuda = True if opt.use_gpu == 1 and torch.cuda.is_available() else False

    trainset, valset = help.ReadData(opt.dataset_name, train_size=opt.data_size, prob=opt.prob)

    ## if save samples, only for low data scenario
    if opt.save_sample:
        os.mkdir(os.path.join(opt.log_dir, 'data'))
        with open(os.path.join(opt.log_dir, 'data','low_sample.pkl'), 'wb') as handle:
            pickle.dump((trainset, valset), handle, protocol=pickle.HIGHEST_PROTOCOL)

    valset = help.torchVariable_from_numpy(valset, float_dtype)
    
    ## construct dataset (dataloader) for training
    train_Dataset = help.Dataset_syn(trainset, opt.output_size)
    assert (trainset.shape[1] - opt.output_size)%2 == 0 # assert an exception if the dimensions are not even for agents

    # --------------- initialization ---------------
    ## initialize checkpoint 
    checkpoint = {
        'metrics_train': [],
        'metrics_val': [],
        'epochs': [],
        'G_loss': [],
        'D_loss': [],
        'g_state': None, 
        'g_optim_state': None,
        'd_state': None,
        'd_optim_state': None,
        'g_best_state': None,
        'd_best_state': None,
        'g_optim_best_state':None,
        'd_optim_best_state':None,
        'best_epoch': None,
        'min_metric': None
    }

    ## initialize D and G
    agent_generator = models.AgentGenerator(opt)
    agent_discriminator = models.AgentDiscriminator(opt)

    if cuda:
        agent_generator.cuda()
        agent_discriminator.cuda()

    ## configure dataloader
    dataloader = DataLoader(
        dataset=train_Dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True
    )

    ## optimizer
    if opt.optimName =='adam':
        optimizer_G = torch.optim.Adam(agent_generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        optimizer_D = torch.optim.Adam(agent_discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    elif opt.optimName =='rmsp':
        optimizer_G = torch.optim.RMSprop(agent_generator.parameters(), lr=opt.lr)
        optimizer_D = torch.optim.RMSprop(agent_discriminator.parameters(), lr=opt.lr)

    # ----------
    #  Training
    # ----------

    for epoch in range(opt.n_epochs):
        # generate new data
        dataloader_iter = iter(dataloader)
        (x_train, _)  = next(dataloader_iter)

        batch_size = x_train.size(0)

        # configure input
        ## transform into variable
        x1_train = Variable(x_train[:, :2].type(float_dtype))

        # ----------
        #  Train Discriminator
        # ----------

        # Sample noise and conditions (structured and timeseries) 
        # as generator input
        noise_shape = (batch_size, opt.latent_dim)
        z = help.get_noise(noise_shape, opt.noise_type, float_dtype) # z is a Variable here

        # Generate a batch of demands
        x1_train_gen = agent_generator(z)

        # validaty for real demand
        validity_real = agent_discriminator(x1_train)

        # validaty for fake demand
        validity_fake = agent_discriminator(x1_train_gen)

        # gradient penalty 
        gradient_penalty = losses.compute_gradient_penalty(
            agent_discriminator, x1_train.data, x1_train_gen.data, 
            float_dtype
        )

        # Adversarial loss
        d_loss = -torch.mean(validity_real) + torch.mean(validity_fake) + lambda_gp*gradient_penalty
        w_loss = -torch.mean(validity_real) + torch.mean(validity_fake)

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # ----------
        #  Train Generator
        # ----------
        if epoch% opt.n_critic ==0:

            # Generate a batch of attributes
            x1_train_gen = agent_generator(z)

            # Loss measures generator's ability to fool the discriminator
            fake_validity = agent_discriminator(x1_train_gen)
            g_loss = -torch.mean(fake_validity)

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

        # print metrics
        if opt.trackTraining=='all':
            if epoch% opt.print_interval == 0:
                print(
                    "[Epoch %d/%d] [W loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, w_loss.item(), g_loss.item())
                )
        

        # --------------- train metrics ---------------
        if epoch % opt.n_record == 0:
            ## conduct the network
            noise_shape = (valset.shape[0], opt.latent_dim)
            z = help.get_noise(noise_shape, opt.noise_type, float_dtype)
            x1_val_gen = agent_generator(z)
            validity_real = agent_discriminator(valset[:, 2:4])
            validity_fake = agent_discriminator(x1_val_gen)
            
            w_loss = -torch.mean(validity_real) + torch.mean(validity_fake) # this is the real wassertein loss which reprents the <minus> wasserstein distance
                                                                            # between real and fake distribution
            g_loss = -torch.mean(validity_fake)

            ## metric
            fake_sample = x1_val_gen.cpu().data if cuda else x1_val_gen.data
            real_sample = valset[:, 2:4].cpu().data if cuda else valset[:, 2:4].data
            metric_train = losses.emd(fake_sample, real_sample)

            if opt.trainingMode=='optuna':
                trial.report(metric_train,epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if opt.trackTraining=='all':
                print(
                    "[Epoch %d/%d] [Train EMD: %f]\n"
                    % (epoch, opt.n_epochs, metric_train)
                )

            if (opt.plotOutput=='agent' or opt.plotOutput=='all') and epoch %1==0:
                help.plotOutputScatter(real_sample, fake_sample, \
                    epoch, metric_train, opt.log_dir)

            checkpoint['metrics_train'].append(metric_train)
            checkpoint['epochs'].append(epoch)
            checkpoint['G_loss'].append(g_loss.item())
            checkpoint['D_loss'].append(w_loss.item())        

            if checkpoint['min_metric'] == None:
                # let current metric be min_metric
                checkpoint['g_best_state'] = deepcopy(agent_generator.state_dict())
                checkpoint['d_best_state'] = deepcopy(agent_discriminator.state_dict())
                checkpoint['g_optim_best_state'] = deepcopy(optimizer_G.state_dict())
                checkpoint['d_optim_best_state'] = deepcopy(optimizer_D.state_dict())
                checkpoint['best_epoch'] = epoch
                checkpoint['min_metric'] = metric_train
            elif metric_train < checkpoint['min_metric']:
                # update metric and network
                checkpoint['g_best_state'] = deepcopy(agent_generator.state_dict())
                checkpoint['d_best_state'] = deepcopy(agent_discriminator.state_dict())
                checkpoint['g_optim_best_state'] = deepcopy(optimizer_G.state_dict())
                checkpoint['d_optim_best_state'] = deepcopy(optimizer_D.state_dict())
                checkpoint['best_epoch'] = epoch
                checkpoint['min_metric'] = metric_train
    
    # ----------
    #  return checkpoint
    # ----------
    checkpoint['g_state'] = deepcopy(agent_generator.state_dict())
    checkpoint['g_optim_state'] = deepcopy(optimizer_G.state_dict())
    checkpoint['d_state'] = deepcopy(agent_discriminator.state_dict())
    checkpoint['d_optim_state'] = deepcopy(optimizer_D.state_dict())

    return checkpoint
