################
# import the packages
################

import os
from copy import deepcopy

import optuna
import torch
from gan import help, losses, models
from torch.autograd import Variable
from torch.utils.data import DataLoader

################
# main
################

def main(opt,timestamp=None, trial=None):
    # --------------- input arguments ---------------
    lambda_gp = opt.lambda_gp # hyperparameter lambda
    long_dtype, float_dtype = help.get_dtypes(opt)

    cuda = True if opt.use_gpu == 1 and torch.cuda.is_available() else False

    trainset, valset = help.ReadData(opt.dataset_name, train_size=opt.data_size, prob=opt.prob)
    
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
    agent_generator=models.AgentGenerator(opt)
    agent_discriminator=models.AgentDiscriminator(opt)

    if cuda:
        agent_generator.cuda()
        agent_discriminator.cuda()

    ## new critic: build a new critic to provide loss/feedback
    sys_critic = models.Sys_critic_mgm(opt)
    critic_optimizer= torch.optim.Adam(sys_critic.parameters(), lr=opt.lrSys, betas=(opt.b1Sys, opt.b2Sys))
    if cuda:
        sys_critic.cuda()

    ## restore sysGAN
    restore_path = os.path.join(opt.checkpoint_dir, opt.ckpt_name)

    if cuda:
        sys_generator = models.SysGenerator(opt)
        sys_discriminator = models.SysDiscriminator(opt)
        checkpoint_sys = torch.load(restore_path)
        sys_generator.load_state_dict(checkpoint_sys['g_best_state'])
        sys_discriminator.load_state_dict(checkpoint_sys['d_best_state'])
        sys_generator.cuda()
        sys_discriminator.cuda()
        sys_optimizerD = torch.optim.Adam(sys_discriminator.parameters(), lr=opt.lrSys, betas=(opt.b1Sys, opt.b2Sys))
        sys_optimizerD.load_state_dict(checkpoint_sys['d_optim_best_state'])
    else:
        sys_generator = models.SysGenerator(opt)
        sys_discriminator = models.SysDiscriminator(opt)
        checkpoint_sys = torch.load(restore_path, map_location = torch.device('cpu'))
        sys_generator.load_state_dict(checkpoint_sys['g_best_state'])
        sys_discriminator.load_state_dict(checkpoint_sys['d_best_state'])
        sys_optimizerD = torch.optim.Adam(sys_discriminator.parameters(), lr=opt.lrSys, betas=(opt.b1Sys, opt.b2Sys))
        sys_optimizerD.load_state_dict(checkpoint_sys['d_optim_best_state'])

    ## freeze the weight of sys generator
    for param in sys_generator.parameters():
        param.requires_grad = False

    ## configure dataloader
    dataloader = DataLoader(
        dataset=train_Dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True
    )

    ## optimizer
    if opt.optimName == 'adam':
        optimizer_G = torch.optim.Adam(agent_generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        optimizer_D = torch.optim.Adam(agent_discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    elif opt.optimName == 'rmsp':
            optimizer_G = torch.optim.RMSprop(agent_generator.parameters(), lr=opt.lr)
            optimizer_D = torch.optim.RMSprop(agent_discriminator.parameters(), lr=opt.lr)
    else:
        raise Exception('Arg [--optimName]: Only "adam" or "rmsp" is allowed')

    # ----------
    #  Training
    # ----------
    for epoch in range(opt.n_epochs):
        for i, (x_train, s_train) in enumerate(dataloader):
            batch_size = x_train.size(0)

            # configure input
            ## transform into variable
            x_train = Variable(x_train.type(float_dtype))
            s_train = Variable(s_train.type(float_dtype))

            # ----------
            #  Train Discriminator
            # ----------
            # Sample noise and conditions (structured and timeseries) 
            # as generator input
            noise_shape = (batch_size, opt.latent_dim)
            z = help.get_noise(noise_shape, opt.noise_type, float_dtype) # z is a Variable here

            # Generate a batch of demands
            x_train_gen = agent_generator(z)

            # validaty for real demand
            validity_real = agent_discriminator(x_train[:,:2])

            # validaty for fake demand
            validity_fake = agent_discriminator(x_train_gen)

            # gradient penalty 
            gradient_penalty = losses.compute_gradient_penalty(
                agent_discriminator, x_train[:,:2].data,\
                    x_train_gen.data, float_dtype
            )

            # Adversarial loss
            d_loss = -torch.mean(validity_real) + torch.mean(validity_fake) + \
                lambda_gp*gradient_penalty
            w_loss = -torch.mean(validity_real) + torch.mean(validity_fake) # wasserstein loss (d_loss excluding gradient penalty)

            optimizer_D.zero_grad()
            optimizer_G.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            # ----------
            #  Train Sys Discriminator
            # ----------
            z = help.get_noise(noise_shape, opt.noise_type, float_dtype)
            x_train_gen_x1 = agent_generator(z)
            s_train_gen_x1 = sys_generator(z, [x_train_gen_x1, x_train[:, 2:4]])
            validity_sys_real = sys_critic(s_train)
            validity_sys_fake = sys_critic(s_train_gen_x1)

            gradient_penalty_sys = losses.compute_gradient_penalty(
                sys_critic, s_train.data, s_train_gen_x1.data, 
                float_dtype
            )

            # Adversarial loss
            d_loss_sys = -torch.mean(validity_sys_real)+torch.mean(validity_sys_fake) + opt.lambdaSys*gradient_penalty_sys

            critic_optimizer.zero_grad()
            optimizer_G.zero_grad()
            d_loss_sys.backward()
            critic_optimizer.step()

            # ----------
            #  Train Generator
            # ----------
            if i% opt.n_critic ==0:
                z = help.get_noise(noise_shape, opt.noise_type, float_dtype)
                # Generate a batch of attributes
                x_train_gen = agent_generator(z)

                # Loss measures generator's ability to fool the discriminator
                validity_fake = agent_discriminator(x_train_gen)

                # Generate system-level data using real agent-level data
                s_train_gen = sys_generator(z, [x_train_gen, x_train[:, 2:4]])
                validity_fake_s = sys_critic(s_train_gen)

                # gnerator loss
                g_loss = -(1-opt.alpha)*torch.mean(validity_fake)\
                    -opt.alpha*torch.mean(validity_fake_s)
                
                critic_optimizer.zero_grad()
                optimizer_G.zero_grad()
                optimizer_D.zero_grad()
                g_loss.backward()

                optimizer_G.step()

            # print metrics
            if opt.trackTraining=='all':
                if i% opt.print_interval == 0:
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [W loss: %f] [G loss: %f]"
                        % (epoch, opt.n_epochs, i, len(dataloader), w_loss.item(), g_loss.item())
                    )
        

        # --------------- train metrics ---------------
        ## conduct the network
        noise_shape = (valset.shape[0], opt.latent_dim)
        z = help.get_noise(noise_shape, opt.noise_type, float_dtype)
        x1_val_gen = agent_generator(z)
        s_val_gen_x1 = sys_generator(z, [x1_val_gen, valset[:, 4:6]])
        validity_real = agent_discriminator(valset[:,2:4])
        validity_fake = agent_discriminator(x1_val_gen)
        validity_sys_fake = sys_critic(s_val_gen_x1)
        
        w_loss = -torch.mean(validity_real) + torch.mean(validity_fake) # this is the real wassertein loss which reprents the <minus> wasserstein distance
                                                                        # between real and fake distribution
        g_loss = -(1-opt.alpha)*torch.mean(validity_fake)\
            -opt.alpha*torch.mean(validity_sys_fake)

        # metric
        fake_sample = x1_val_gen.cpu().data if cuda else x1_val_gen.data
        real_sample = valset[:, 2:4].cpu().data if cuda else valset[:, 2:4].data
        metric_train = losses.emd(fake_sample, real_sample)

        if opt.trainingMode=='optuna':
            trial.report(metric_train, epoch)
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
