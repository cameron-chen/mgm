from gan import models
from gan import help
from gan import losses
import torch
import argparse
import os
import re
import pickle

import matplotlib.pyplot as plt
import numpy as np

# load the model and real data
def G_model(restore_path, opt):
    G_a = models.AgentGenerator(opt)
    ckpt= torch.load(restore_path)
    G_a.load_state_dict(ckpt['g_best_state'])
    # print(G_a)

    return G_a

def z_scaler(sample, mean, std):
    return (sample*std+mean)

# task1: viz
## generate sample
def generate_sample(G, shape, noise_type='gaussian'):
    z = help.get_noise(shape, noise_type, torch.FloatTensor)
    x = G(z).detach().numpy()

    return x

## plot and save the results
def single_plot(x_fake, x_real, root_path, metric=None, file_name=None, sample_size=None):
    if sample_size:
        idx = np.random.permutation(x_fake.shape[0])[:sample_size]
    else:
        idx = np.arange(x_fake.shape[0])
    plt.figure(figsize=(6.4,6.4), dpi=100)
    plt.scatter(x=x_real[idx,0],y=x_real[idx,1],s=15,marker='x',color='#E3B324',label='Real',alpha=0.8)
    plt.scatter(x=x_fake[idx,0],y=x_fake[idx,1],s=15,marker='x',color='#256F1F',label='Fake',alpha=1.0)
    plt.legend()
    if metric:
        plt.title('Metric: {:.3f}'.format(metric))

    file_path = os.path.join(root_path, file_name) if file_name else os.path.join(root_path, 'viz')
    plt.savefig(file_path)
    plt.close('all')
    print('Save the plot at {}'.format(file_path))

# task2: multiple runs (default: 16)
def multi_eva(G, valset, num_run=16, shape=None, stat=None, **kwargs):
    metrics = list()
    if stat is not None:
        # recover normalized data
        mean_scaler = stat['mean']
        std_scaler = stat['std']
        for i in range(num_run):
            x_fake = generate_sample(G, shape)
            x_fake = z_scaler(x_fake, mean_scaler, std_scaler)
            x_real = valset
            metrics.append(losses.emd(x_fake, x_real))
            if i == 0:
                single_plot(x_fake,x_real,**kwargs)
    else:
        for i in range(num_run):
            x_fake = generate_sample(G, shape)
            x_real = valset
            metrics.append(losses.emd(x_fake, x_real))
            if i == 0:
                single_plot(x_fake,x_real,**kwargs)
    
    return metrics

def record_eva(root_path, metrics, file_name = 'metrics.txt'):
    log_path = os.path.join(root_path, file_name)
    metric_mean, metric_std = np.mean(metrics), np.std(metrics)

    with open(log_path, 'w') as f:
        f.write('Metric: Wasserstein Distance-1\n')
        f.write(' - Runs: {}\n'.format(len(metrics)))
        f.write(' - Stat: {:.3f}({:.3f})\n\n'.format(metric_mean, metric_std))
        f.write(' - Details: {}\n'.format(', '.join([str(round(float(m),5)) for m in metrics])))
        f.write(' - Possible stat: {:.3f}({:.4f}), {:.4f}({:.4f})\n'.format(metric_mean, metric_std, metric_mean, metric_std))
    
    print('Save the result at {}'.format(log_path))

def main(opt):
    '''Evaluate the model in multiple runs and provide report
    '''
    # load the model and data
    G = G_model(opt.restore_path, opt)
    np.random.seed(opt.seed_data)
    _, valset = help.ReadData(opt.dataset_name, train_size=opt.data_size)

    # multiple runs
    shape = (valset.shape[0], opt.latent_dim)
    config = {
        'root_path': os.path.dirname(opt.restore_path)
    }
    metrics = multi_eva(G, valset, shape=shape, **config)

    # report
    record_eva(os.path.dirname(opt.restore_path), metrics)
    
    return

def multi_cus(opt):
    '''Evaluate the model in multiple runs and save the report in 
    the customized path
    '''
    # load the model and data
    G = G_model(opt.restore_path, opt)
    np.random.seed(opt.seed_data)
    _, valset = help.ReadData(opt.dataset_name, train_size=opt.data_size)

    # multiple runs
    shape = (valset.shape[0], opt.latent_dim)
    config = {
        'root_path': opt.custom_root_path,
        'file_name': opt.custom_file_name
    }
    metrics = multi_eva(G, valset, shape=shape, **config)

    # report
    record_eva(opt.custom_root_path,metrics, file_name=opt.custom_file_name+'.txt')

    return

def optuna_extract_models(trials_best, optuna_study, top_n):
    """extract models to be evaluated from optuna study
    """
    model_list = list()

    for i in range(top_n):
        trialName = trials_best[i]
        trialNum = int(re.search(r'\d+', trialName).group(0))
        Trial = optuna_study.trials[trialNum]
        model_list.append(Trial.user_attrs['ckpt_path'])
    
    return model_list

def optuna_eva(trials_best, optuna_study, study_path, opt, logger, top_n=2, num_eva=16):
    """evaluate the first n trials in optuna

    Args:
        trials_best: a list of top 5 trials' names
        optuna_study: an instance of the optuna study
        study_path: path to the study folder
        opt: hyperparameters
        logger: logger instance
        top_n: top N models which will be evaluated
        num_eva: number of multiple samples from the generator
    """
    # extract model path
    model_list = optuna_extract_models(trials_best, optuna_study, top_n)

    # load the data
    restore_path_lowdataset = os.path.join(
        opt.restore_path_lowdataset, 'data/low_sample.pkl'
    )
    with open('{}'.format(restore_path_lowdataset),'rb') as handle:
        _,valset = pickle.load(handle)
    input_shape = (valset.shape[0], opt.latent_dim)

    # multiple runs
    logger.info('-- Evaluation --')
    logger.info('Metric: Wasserstein Distance-1')
    for model_name in model_list:
        # recover generator
        restore_path_model = model_name
        G = G_model(restore_path_model, opt)

        # multiple runs
        config = {
            'root_path':os.path.join(study_path, 'plot'),
            'file_name':'{}-generated-sample'.format(os.path.basename(model_name)[:3])}
        metrics = multi_eva(G, valset[:, 2:4], num_run=num_eva, shape=input_shape,**config)
        metric_mean, metric_std = np.mean(metrics), np.std(metrics)

        # report
        logger.info(' - Model name: {}'.format(os.path.basename(model_name)))
        logger.info(' - Runs: {}'.format(len(metrics)))
        logger.info(' - Stat: {:.3f}({:.3f})'.format(metric_mean, metric_std))
        logger.info(' - Details: {}'.format(', '.join([str(round(float(m),5)) for m in metrics])))
        logger.info(' - Possible stat: {:.3f}({:.4f}), {:.4f}({:.4f})\n'.format(metric_mean, metric_std, metric_mean, metric_std))
    return

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_size_agent", type=int, default=2, help="size of agent generator output")
    parser.add_argument("--latent_dim", type=int, default=2, help="dimensionality of the latent space")
    parser.add_argument("--restore_path", type=str, default='', help="restore path of the checkpoint")
    parser.add_argument("--dataset_name", type=str, default='data_roll_b_0.5.csv', help="file name of the dataset")
    parser.add_argument("--data_size", type=int, default=128000, help="the size of the data used for experiment")
    parser.add_argument("--sample_size", type=int, default=100, help="the size of the sample")
    parser.add_argument("--mode", type=str, choices=['multi', 'multi-cus'])
    parser.add_argument("--custom_root_path", type=str, default='')
    parser.add_argument("--custom_file_name", type=str, default='')
    parser.add_argument("--seed_data", type=int, default=46, help="seed used for sampling data, especially low data scenario")
    opt= parser.parse_args()

    if opt.mode == 'multi':
        # evaluate: save in the default path
        main(opt)
    elif opt.mode == 'multi-cus':
        # evaluate: save in the customized path
        assert all(n!='' for n in [opt.custom_root_path, opt.custom_file_name])

        multi_cus(opt)
