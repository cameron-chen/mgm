import argparse
import datetime
import os
from copy import deepcopy
from statistics import mean

import numpy as np
import optuna
import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from util import data_util, gan_util, help_fn, losses

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"   
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main(opt, timestamp = None, trial = None):
    # input arguments
    long_dtype, float_dtype = help_fn.get_dtypes(bool(opt.use_gpu))
    cuda = True if bool(opt.use_gpu) and torch.cuda.is_available() else False

    batch_size = opt.batch_size
    batch_size_val = opt.batch_size_val
    time_steps = opt.time_steps
    z_dims_t = opt.z_dims_t
    y_dims = opt.Dy
    Dx = opt.Dx
    cond_size = opt.cond_size
    noise_type = opt.noise_type

    sinkhorn_eps = opt.sinkhorn_eps
    sinkhorn_l = opt.sinkhorn_l
    scaling_coef = 1.0
    disc_iters = 1

    # load data
    if opt.dname == 'AROne':
        no = int(1e4)
        Dx = 10 
        train_data = data_util.AROne_data_generation(
            no, Dx, time_steps, np.linspace(0.1, 0.9,Dx), 0.5)
        val_data = train_data
    elif opt.dname == 'Elec':
        train_data = data_util.real_data_loading(
            opt.dname, seq_len=time_steps, trunc_head_perc=opt.thp,
            stride=opt.stride, nor_method='z_score')
        val_data = data_util.real_data_loading(
            opt.dname, seq_len=time_steps,
            stride=opt.stride_val, nor_method='z_score')
    elif opt.dname == 'Elec_low':
        train_data = data_util.real_data_loading(
            opt.dname, seq_len=time_steps,
            stride=opt.stride, nor_method='z_score')
        val_data = data_util.real_data_loading(
            opt.dname, seq_len=time_steps,
            stride=opt.stride_val, nor_method='z_score')
        train_data = train_data[:opt.low_data_size].reshape(opt.low_data_size, time_steps, -1)
    else:
        ValueError('Dataset does not exist.')

    dataset = data_util.Dataset_full_data(train_data)
    dataset_val = data_util.Dataset_full_data(val_data)
    loader = DataLoader(dataset, batch_size=batch_size*2, drop_last=True)

    # initialization
    ## net
    ### model
    g_state_size = opt.g_state_size
    d_state_size = opt.d_state_size
    g_filter_size = opt.g_filter_size
    d_filter_size = opt.d_filter_size
    nlstm = opt.nlstm
    bn = bool(opt.bn)
    cond_disc = bool(opt.cond_disc)
    disc_kernel_width = 5
    g_output_activation = opt.g_output_activation

    gen_lr = opt.lr
    disc_lr = opt.lr


    G = gan_util.Sys_Generator(
        time_steps, z_dims_t, Dx, g_state_size, g_filter_size,
        output_activation=g_output_activation, nlstm=nlstm, nlayer=2,
        Dy=y_dims, bn=bn, cond_size=cond_size
        )
    #* note, the nlstm and nlayer are useless for generator as it is fixed
    #* check the code for details
    D_h = gan_util.Sys_Discriminator(
        time_steps, z_dims_t, Dx, d_state_size, d_filter_size,
        kernel_size=disc_kernel_width, nlayer=2, nlstm=0, bn=bn,
        cond_disc=cond_disc, cond_size=cond_size
        )
    D_m = gan_util.Sys_Discriminator(
        time_steps, z_dims_t, Dx, d_state_size, d_filter_size,
        kernel_size=disc_kernel_width, nlayer=2, nlstm=0, bn=bn,
        cond_disc=cond_disc, cond_size=cond_size
        )
    if cuda:
        G.cuda(), D_h.cuda(), D_m.cuda()
    ### optimizer
    Optim_G = optim.Adam(G.parameters(), lr=gen_lr, betas=(opt.b1, opt.b2))
    Optim_D_h = optim.Adam(D_h.parameters(), lr=disc_lr, betas=(opt.b1, opt.b2))
    Optim_D_m = optim.Adam(D_m.parameters(), lr=disc_lr, betas=(opt.b1, opt.b2))

    ## record
    ### checkpoint
    checkpoint = {
        'metric': [],
        'iter': [],
        'G_loss': [],
        'g_state': None, 
        'g_optim_state': None,
        'd_h_state': None,
        'd_h_optim_state': None,
        'd_m_state': None,
        'd_m_optim_state': None,
        'best_iter': None,
        'min_metric': None,
        'g_state_best': None, 
        'g_optim_state_best': None,
        'd_h_state_best': None,
        'd_h_optim_state_best': None,
        'd_m_state_best': None,
        'd_m_optim_state_best': None,
        'min_metric_rep': None,
    }

    ### tensorboard logs
    if bool(opt.tb_log): # tb_log: tensorboard logs
        _test  = opt.dname + '_cond_cot'

        save_file = "{}_{}".format(_test,timestamp.strftime("%h%d-%H:%M:%S.%f"))

        log_dir = "./output/{}/log".format(save_file)

        # create dirctories for storing samples later
        if not os.path.exists('./output/{}/data'.format(save_file)):
            os.makedirs('./output/{}/data'.format(save_file))
        if not os.path.exists('./output/{}/sample'.format(save_file)):
            os.makedirs('./output/{}/sample'.format(save_file))

        # GAN train notes 
        with open("./output/{}/train_notes.txt".format(save_file), 'w') as f:
            # Include any experiment notes here:
            f.write("Experiment notes: .... \n\n")
            f.write("*Note:\n{}\n\n".format(opt.note))
            f.write("MODEL_DATA: {}\nSEQ_LEN: {}\n".format(
                opt.dname,
                time_steps, ))
            f.write("STATE_SIZE: {}\nNUM_LAYERS_LSTM: {}\nLAMBDA: {}\n".format(
                g_state_size,
                nlstm,
                opt.reg_penalty))
            f.write("BATCH_SIZE: {}\nCRITIC_ITERS: {}\nGenerator LR: {}\nDiscriminator LR:{}\n".format(
                batch_size,
                disc_iters,
                gen_lr,
                disc_lr))
            f.write("SINKHORN EPS: {}\nSINKHORN L: {}\nG_OUTPUT_ACTIVATION: {}\n".format(
                sinkhorn_eps,
                sinkhorn_l,
                g_output_activation))
            if opt.dname == 'Elec_low':
                f.write("LOW DATA SIZE: {}\n".format(opt.low_data_size))
            f.write("CONDITIAL DISC: {}\n\n".format(
                cond_disc
                ))

        # record save_file to trial.user_attribute
        if trial:
            trial.set_user_attr('save_file', save_file)

        writer = SummaryWriter(log_dir)

    # one step training func
    def disc_training_step(real_data, real_data_p, condition, condition_p, cond_disc=True):
        """
        Args:
            cond_disc: bool, conditional discrimintator
        """
        hidden_z = help_fn.get_noise([batch_size, time_steps, z_dims_t],float_dtype, noise_type=noise_type)
        hidden_z_p = help_fn.get_noise([batch_size, time_steps, z_dims_t],float_dtype, noise_type=noise_type)
        hidden_y = help_fn.get_noise([batch_size, y_dims],float_dtype, noise_type=noise_type)
        hidden_y_p = help_fn.get_noise([batch_size, y_dims],float_dtype, noise_type=noise_type)

        fake_data = G(hidden_z, condition, hidden_y)
        fake_data_p = G(hidden_z_p, condition_p, hidden_y_p)
        #* fake_data.shape: (batch_size, time_steps, Dx)
        #* fake_data refers to system output here

        if cond_disc:
            real_data = torch.cat([condition, real_data], -1)
            real_data_p = torch.cat([condition_p, real_data_p], -1)
            fake_data = torch.cat([condition, fake_data], -1)
            fake_data_p = torch.cat([condition_p, fake_data_p], -1)

        h_fake = D_h(fake_data)

        m_real = D_m(real_data)
        m_fake = D_m(fake_data)

        h_real_p = D_h(real_data_p)
        h_fake_p = D_h(fake_data_p)

        m_real_p = D_m(real_data_p)

        loss1 = losses.compute_mixed_sinkhorn_loss(
            real_data, fake_data, m_real, m_fake, h_fake,
            sinkhorn_eps, sinkhorn_l, real_data_p, fake_data_p, m_real_p,
            h_real_p, h_fake_p
        )
        pm1 = losses.scale_invariante_martingale_regularization(
            m_real, opt.reg_penalty
        )
        disc_loss = -loss1+pm1

        # update discriminator parameters
        Optim_D_h.zero_grad()
        Optim_D_m.zero_grad()
        disc_loss.backward()
        Optim_D_h.step()
        Optim_D_m.step()

    def gen_training_step(real_data, real_data_p, condition, condition_p, cond_disc=True):
        """
        Args:
            cond_disc: bool, conditional discrimintator
        """
        hidden_z = help_fn.get_noise([batch_size, time_steps, z_dims_t],float_dtype, noise_type=noise_type)
        hidden_z_p = help_fn.get_noise([batch_size, time_steps, z_dims_t],float_dtype, noise_type=noise_type)
        hidden_y = help_fn.get_noise([batch_size, y_dims],float_dtype, noise_type=noise_type)
        hidden_y_p = help_fn.get_noise([batch_size, y_dims],float_dtype, noise_type=noise_type)

        fake_data = G(hidden_z, condition, hidden_y)
        fake_data_p = G(hidden_z_p, condition_p, hidden_y_p)
        #* fake_data.shape: (batch_size, time_steps, Dx)
        #* fake_data refers to system output here

        if cond_disc:
            real_data = torch.cat([condition, real_data], -1)
            real_data_p = torch.cat([condition_p, real_data_p], -1)
            fake_data = torch.cat([condition, fake_data], -1)
            fake_data_p = torch.cat([condition_p, fake_data_p], -1)

        h_fake = D_h(fake_data)

        m_real = D_m(real_data)
        m_fake = D_m(fake_data)

        h_real_p = D_h(real_data_p)
        h_fake_p = D_h(fake_data_p)

        m_real_p = D_m(real_data_p)

        loss2 = losses.compute_mixed_sinkhorn_loss(
            real_data, fake_data, m_real, m_fake, h_fake,
            sinkhorn_eps, sinkhorn_l, real_data_p, fake_data_p, m_real_p,
            h_real_p, h_fake_p
        )
        gen_loss = loss2

        # update generator parameters
        Optim_G.zero_grad()
        gen_loss.backward()
        Optim_G.step()

        return gen_loss


    # training
    for _iter in range(1, opt.n_iter):
        # sample a batch of data
        try:
            x = next(_loader)
        except:
            _loader = iter(loader)
            x = next(_loader)

        x = Variable(x.type(float_dtype))
        real_data = x[:batch_size,:, 3].reshape(batch_size, time_steps, -1)
        real_data_p = x[batch_size:,:, 3].reshape(batch_size, time_steps, -1)
        condition = x[:batch_size,:, :3]
        condition_p = x[batch_size:,:, :3]

        # training step
        disc_training_step(real_data, real_data_p, condition, condition_p, cond_disc=cond_disc)
        loss = gen_training_step(real_data, real_data_p, condition, condition_p, cond_disc=cond_disc)

        # print and record
        if bool(opt.tb_log):
            writer.add_scalar('Sinkhorn training loss', loss, global_step=_iter)
            writer.flush()

        if torch.isinf(loss):
            print('{} Loss exploded'.format(_test))
            # open the training notes with mode a - append
            with open('./output/{}/train_notes.txt'.format(save_file), 'a') as f:
                f.write("\nTraining failed!  ")
            break
        else:
            if _iter % opt.save_freq == 0 or _iter == 1:
                metrics = list()
                for _ in range(5):
                    hidden_z = help_fn.get_noise([batch_size_val, time_steps, z_dims_t],float_dtype, noise_type=noise_type)
                    hidden_y = help_fn.get_noise([batch_size_val, y_dims],float_dtype, noise_type=noise_type)

                    x = dataset_val[np.random.permutation(len(dataset_val))[:batch_size_val]]
                    x = torch.from_numpy(x)
                    x = Variable(x.type(float_dtype))
                    real_data = x[:,:,3].reshape(batch_size_val, time_steps, -1)
                    condition = x[:,:,:3]
                    samples = G(hidden_z, condition, hidden_y)
                    # compute metric
                    metric = losses.abs_autocorr(samples.cpu().data, real_data.cpu().data) if cuda else\
                        losses.abs_autocorr(samples.data, real_data.data)
                    metrics.append(metric)
                
                # save model to file
                checkpoint['metric'].append(mean(metrics).item())
                checkpoint['iter'].append(_iter)
                checkpoint['G_loss'].append(loss)
                checkpoint['g_state'] = deepcopy(G.state_dict())
                checkpoint['g_optim_state'] = deepcopy(Optim_G.state_dict())
                checkpoint['d_h_state'] = deepcopy(D_h.state_dict())
                checkpoint['d_h_optim_state'] = deepcopy(Optim_D_h.state_dict())
                checkpoint['d_m_state'] = deepcopy(D_m.state_dict())
                checkpoint['d_m_optim_state'] = deepcopy(Optim_D_m.state_dict())
                if __name__ == "__main__" or opt.training_mode in ['single_run']:
                    print("iter: {}/{}, sink_dis: {:.5f}, abs_corr: {:.5f}".format(
                        _iter, opt.n_iter, loss.item(), mean(metrics).item()))

                if __name__!="__main__" and opt.training_mode in ['optuna']:
                    trial.report(mean(metrics).item(), _iter)
                    if opt.pruner and trial.should_prune():
                        raise optuna.TrialPruned()

                if checkpoint['min_metric'] is None or checkpoint['min_metric']>mean(metrics).item():
                    # save the model of best status
                    checkpoint['best_iter'] = _iter
                    checkpoint['min_metric'] = mean(metrics).item()
                    checkpoint['g_state_best'] = deepcopy(G.state_dict())
                    checkpoint['g_optim_state_best'] = deepcopy(Optim_G.state_dict())
                    checkpoint['d_h_state_best'] = deepcopy(D_h.state_dict())
                    checkpoint['d_h_optim_state_best'] = deepcopy(Optim_D_h.state_dict())
                    checkpoint['d_m_state_best'] = deepcopy(D_m.state_dict())
                    checkpoint['d_m_optim_state_best'] = deepcopy(Optim_D_m.state_dict())
                    checkpoint['min_metric_rep'] = deepcopy(metrics) 

                if bool(opt.tb_log):
                    torch.save(checkpoint,"./output/{}/ckpt.pt".format(save_file))
                    writer.add_scalar('Absolute Difference of Autocorrelation', mean(metrics).item(), global_step=_iter)
                    writer.flush()

    if bool(opt.tb_log):
        end_time = help_fn.datetime_as_timezone(datetime.datetime.utcnow())
        with open('./output/{}/train_notes.txt'.format(save_file), 'a') as f:
            f.write('\nTraining start from: {}'.format(timestamp.strftime("%h%d-%H:%M:%S.%f")))
            f.write('\nTraining end at: {}'.format(end_time.strftime("%h%d-%H:%M:%S.%f")))
            f.write('\n\nResult:')
            f.write('\n - Min metric: {:.3f}({:.3f}) at iter {}'.format(
                checkpoint['min_metric'], 
                np.std(checkpoint['min_metric_rep']),
                checkpoint['best_iter']
            ))
            f.write('\n - Metric replica: {}'.format(", ".join([str(round(x,3)) for x in checkpoint['min_metric_rep']])))
        
        writer.close()

    return checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cot')
    parser.add_argument('-d', '--dname', type=str, default='Elec',
                        choices=['AROne', 'Elec', 'Elec_low'])
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-gss', '--g_state_size', type=int, default=32)
    parser.add_argument('-dss', '--d_state_size', type=int, default=32)
    parser.add_argument('-gfs', '--g_filter_size', type=int, default=32)
    parser.add_argument('-dfs', '--d_filter_size', type=int, default=32)
    parser.add_argument('-goa', '--g_output_activation', type=str, default='linear',
                        choices=['linear', 'tanh'])
    parser.add_argument('-r', '--reg_penalty', type=float, default=10.0)
    parser.add_argument('-ts', '--time_steps', type=int, default=48)
    parser.add_argument('-stride', '--stride', type=int, default=1)
    parser.add_argument('-s_val', '--stride_val', type=int, default=1)
    parser.add_argument('-sinke', '--sinkhorn_eps', type=float, default=100)
    parser.add_argument('-sinkl', '--sinkhorn_l', type=int, default=100)
    parser.add_argument('-Dx', '--Dx', type=int, default=1)
    parser.add_argument('-Dy', '--Dy', type=int, default=10)
    parser.add_argument('-Dz', '--z_dims_t', type=int, default=10)
    parser.add_argument('-Dc', '--cond_size', type=int, default=3)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-bs_val', '--batch_size_val', type=int, default=5000)
    parser.add_argument('-lds', '--low_data_size', type=int, default=64)
    parser.add_argument('-thp', '--thp', type=float, default=0,
                    help="percentage of truncate head of original data")
    parser.add_argument('-nlstm', '--nlstm', type=int, default=1,
                        help="number of lstms in discriminator")
    parser.add_argument('-lr', '--lr', type=float, default=1e-3)
    parser.add_argument('-b1', '--b1', type=float, default=0.9)
    parser.add_argument('-b2', '--b2', type=float, default=0.999)
    parser.add_argument('-bn', '--bn', type=int, default=1,
                        help="batch norm")
    parser.add_argument('-n_iter', '--n_iter', type=int, default=100000)
    parser.add_argument('-save_freq', '--save_freq', type=int, default=100)
    parser.add_argument('-use_gpu', '--use_gpu', type=int, default=1,
                        choices=[0,1])
    parser.add_argument('-noise', '--noise_type', type=str, default='uniform',
                        choices=['gaussian','uniform'], help='distribution of random noise')
    parser.add_argument('-cond_disc', '--cond_disc', type=int, default=1,
                        choices=[0, 1], help='conditional discrminator')
    parser.add_argument('-log', '--tb_log', type=int, default=0,
                        choices=[0, 1], help='tensorboard logs')
    parser.add_argument('-note', '--note', type=str, default='-')

    opt = parser.parse_args()

    timestamp = help_fn.datetime_as_timezone(datetime.datetime.utcnow())

    if bool(opt.seed):
        help_fn.seed_torch(opt.seed)

    main(opt, timestamp)
