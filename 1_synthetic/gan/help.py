import json
import os
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import plotly
import plotly.io as pio
import torch
from optuna.trial import TrialState
from pytz import timezone
from torch.autograd import Variable
from torch.utils.data import Dataset


## choose cup tensor or gpu tensor
def get_dtypes(opt):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if opt.use_gpu == 1 and torch.cuda.is_available():
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype

## choose noise type: Gaussian or Uniform
def get_noise(shape, noise_type, float_dtype):
    '''
    generate normal noise or uniform noise

    Args:
        shape -- the shape of generated noise vector
        noise_type -- 'gaussian' or 'uniform' 

    Returns:
        (noise) -- a tensor of noise
    '''
    if noise_type == 'gaussian':
        return Variable(float_dtype(np.random.normal(0,1, shape)))
    elif noise_type == 'uniform':
        return Variable(float_dtype(np.random.uniform(-1, 1, shape))) # Uni distribution on [-1,1)
    raise ValueError('Unrecognized noise type "%s"' % noise_type)

## get Singapore DateTime
def datetime_as_timezone(date_time, time_zone = 'Asia/Singapore'):
    tz = timezone(time_zone)
    utc = timezone('UTC')
    return date_time.replace(tzinfo=utc).astimezone(tz)

## logs
def num_currnt_run(folder_path, num_digit=3):
    subfolders = [sf for sf in os.listdir(folder_path) if \
        os.path.isdir(os.path.join(folder_path,sf))]
    
    if len(subfolders)==0:
        return 0
    else:
        return max([int(sf[:num_digit]) for sf in subfolders])+1

def init_singlerun(training_object, timestamp):
    '''Initialize single run, create the folder to save logs

    Args:
        training_object: type of the model
        timestamp

    Returns:
        dir of the new folder
    '''
    if not os.path.exists('./output/singlerun/'):
        os.makedirs('./output/singlerun/')
        os.mkdir('./output/singlerun/agent')
        os.mkdir('./output/singlerun/mixer')
        os.mkdir('./output/singlerun/mgm')
    
    if 'agent' in training_object:
        folder_path = './output/singlerun/agent'
    elif 'sys' in training_object:
        folder_path = './output/singlerun/mixer'
    elif 'hier' in training_object:
        folder_path = './output/singlerun/mgm'

    num_current_run = num_currnt_run(folder_path)
    new_sub_folder = '{:03d}-{:s}'.format(
        num_current_run, timestamp.strftime("%b%d-%H:%M:%S"))
    os.mkdir(os.path.join(folder_path, new_sub_folder))

    return os.path.join(folder_path, new_sub_folder)

def save_conf(dir_save, opt):
    '''save configurations
    '''
    with open(os.path.join(dir_save, 'conf.txt'), 'w') as f:
        f.write('Configuration: \n')
        for k, v in vars(opt).items():
            f.write('{:>25}   {}\n'.format(k, v))

## plot loss
def plot_loss(checkpoint, log_dir):
    '''This function is used for plot the loss (d_loss, g_loss, and Metrics) for model

    Args:
        checkpoint
        log_dir
    '''
    g_loss = checkpoint['G_loss']
    d_loss = checkpoint['D_loss']
    metric_train = checkpoint['metrics_train']
    epochs = checkpoint['epochs']

    # plot the graph - procedure
    plt.figure(figsize=(20,10))
    ## fig1_up: D_loss
    plt.subplot(2,3,1)
    plt.plot(epochs, d_loss)
    plt.ylabel("D_loss")
    ## fig1_down: G_loss
    plt.subplot(2,3,4)
    plt.plot(epochs, g_loss)
    plt.xlabel("Epochs")
    plt.ylabel("G_loss")
    ## fig2: Train Metric
    plt.subplot(1,3,(2,3))
    plt.plot(epochs, metric_train)
    plt.ylabel("Train EMD")
    plt.xlabel("Epochs")

    # save the plot
    plt.savefig(os.path.join(log_dir, 'loss_metrics.png'))

    return

def plotOutputScatter(trueData, fakeData, epoch, emd, dir_save):
    '''plot both trueData and fakeData on one figure
    '''
    title = 'Epoch: '+str(epoch)+'   EMD: '+str(emd)
    
    if trueData.size(1)==1:
        true = trueData.detach().numpy().reshape(trueData.size(0))
        fake = fakeData.detach().numpy().reshape(fakeData.size(0))
        # input with size (batch_size, 1)
        plt.hist(true,label='Ground truth',color='red',alpha=0.5, bins=100)
        plt.hist(fake,label='Generated data',color='blue',alpha=0.5, bins=100)
    elif trueData.size(1)==2:
        # input with size (batch_size, 2)
        plt.scatter(trueData[:,0], trueData[:,1],label='Ground truth',color='red', s=4)
        plt.scatter(fakeData[:,0], fakeData[:,1], label='Generated data',color='blue', s=4)
    
    plt.legend()
    plt.title(title)

    # creaete folder
    if not os.path.exists(os.path.join(dir_save, 'sample_plot')): 
        os.mkdir(os.path.join(dir_save, 'sample_plot'))
    # save image
    imgName = 'Epoch_'+str(epoch)
    plt.savefig(os.path.join(dir_save, 'sample_plot', imgName))
    plt.close('all')
    return

## save model 
def save_model(checkpoint, log_dir):
    return torch.save(checkpoint, os.path.join(log_dir, 'ckpt.pt'))

## convert a numpy array to torch Variable
def torchVariable_from_numpy(arr,datatype):
    '''
    convert a numpy array to torch Variable
    '''
    return Variable(torch.from_numpy(arr).type(datatype)) 

# Extract information of Optuna models
def extrResultsOptuna(study, logger):
    '''
    extract the useful information of Optuna model (named study)
    '''

    logger.info("Study statistics: ")
    logger.info("  Number of finished trials: %d" %len(study.trials))
    logger.info("  Number of pruned trials: %d" %len(study.get_trials(deepcopy=False, states=[TrialState.PRUNED])))
    logger.info("  Number of complete trials: %d" %len(study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])))

    logger.info("Best trial:")
    trial = study.best_trial

    logger.info("  Value (avg of last 15 epochs): %.5f" %trial.value)

    logger.info("  Trial Num: %d" %trial.number)

    try:
        trial.user_attrs['manual seed']
    except:
        logger.info("  Seed: no seed recorded in this study instance")
    else:
        logger.info("  Seed: %d" %trial.user_attrs['manual seed'])

    try:
        trial.user_attrs['ckpt_path']
    except:
        logger.info("  Model: no model path recorded in this study instance")
    else:
        logger.info("  Model: {}".format(trial.user_attrs['ckpt_path']))

    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info("    {}: {}".format(key, value))

def exportPlotToPdOptuna(study):
    plotObj = optuna.visualization.plot_intermediate_values(study)
    json_raw = pio.to_json(plotObj)
    json_parsed = json.loads(json_raw)
    layout = json_parsed['layout']
    data_parsed = json_parsed['data']

    df = pd.DataFrame(columns=['name','x','y'])

    for i in range(len(data_parsed)):
        trialInter = data_parsed[i]
        dfTrial = pd.DataFrame({
            'name':trialInter['name'],
            'x':trialInter['x'],
            'y':trialInter['y']
        })

        df = pd.concat([df, dfTrial], ignore_index=True)

    return df, layout

def selectBestResultOptuna(df):
    trialsBest = []
    indexTop50 = df['y'].nsmallest(50).index # top 50 smallest metrics

    for i in indexTop50:
        trialName = df.at[i, 'name']
        if trialName not in trialsBest:
            trialsBest.append(trialName)
        if len(trialsBest) >4:
            break
    
    # return the data top 5 trials containing smallest metric values
    # dfTop5Trials = df[df['name'].isin(trialsBest)]

    # [rectify] return the top 5 trials

    return trialsBest

def reportOptuna(trialsBest, df, plotlyLayout, logger,study, fileName='default'):
    dfTop5Trials = df[df['name'].isin(trialsBest)]

    # plot plotly
    fig = plotOptuna(dfTop5Trials, plotlyLayout, trialsBest)
    # export plotly to png and log the file path
    exportPlotOptuna(fig, logger, fileName)
    # record the trial attributes respectively
    logger.info('-- Trial Details --')
    for trialName in trialsBest:
        dfTrial = df[df['name']==trialName]
        seriesTop5 = [i for i in dfTrial['y'].nsmallest(5)]
        trialNum = int(re.search(r'\d+', trialName).group(0))
        Trial = study.trials[trialNum]
        logger.info('The trial ID is: %s' %trialName)
        logger.info('The top 5 smallest metric values are: ['+\
            ', '.join(str(round(e,3)) for e in seriesTop5)+']')
        logger.info('Trial details:')
        logger.info("  Avg value (avg of last 15 epochs): %.5f" %Trial.value)
        try:
            Trial.user_attrs['manual seed']
        except:
            logger.info("  Seed: no seed recorded in this study instance")
        else:
            logger.info("  Seed: %d" %Trial.user_attrs['manual seed'])
        try:
            Trial.user_attrs['ckpt_path']
        except:
            logger.info("  Model: no model path recorded in this study instance")
        else:
            logger.info("  Model: {}".format(Trial.user_attrs['ckpt_path']))
        logger.info("  Params: ")
        for key, value in Trial.params.items():
            logger.info("    {}: {}".format(key, value))
        logger.info('\n')

    # log: record the trial information

    return pio.to_json(fig)

def plotOptuna(df, layout, trialsBest):
    traces = []
    for trialName in trialsBest:
        dfTrial = df[df['name']==trialName]
        trace = plotly.graph_objects.Scatter(
            x = dfTrial['x'],
            y = dfTrial['y'],
            mode="lines+markers",
            marker={"maxdisplayed": 10},
            name=trialName,
        )
        traces.append(trace)

    fig = plotly.graph_objects.Figure(data=traces, layout=layout)
    fig.update_layout(showlegend=True)
    
    return fig

def exportPlotOptuna(figure, logger, fileName='default'):
    """export optuna plot and log on the logger

    Args:
        fileName: a complete path to save the plot
    """
    filePath = fileName
    pio.write_image(figure, filePath)

    # log: successfully write the top 5 trials image
    logger.info('\n')
    logger.info('-- Save Plot --')
    logger.info('Top 5 trials (png form) have been saved in: <%s>' %filePath)
    # log: record the dir of optuna (in study) and png
    logger.info('Top 5 trials (Plotly object, Json form) would be stored in optuna.study.user_attrs\n'+\
        'Please use <Plotly.io.from_json> to recover\n')

def init_optuna_study(training_object, timestamp):
    '''create a new folder to log optuna study

    Args:
        training_object: type of the model
        timestamp

    Returns:
        dir of the new folder
    '''
    if not os.path.exists('./output/optuna'):
        os.makedirs('./output/optuna')
        os.mkdir('./output/optuna/agent')
        os.mkdir('./output/optuna/mixer')
        os.mkdir('./output/optuna/mgm')
    
    if 'agent' in training_object:
        folder_path = './output/optuna/agent'
    elif 'sys' in training_object:
        folder_path = './output/optuna/mixer'
    elif 'hier' in training_object:
        folder_path = './output/optuna/mgm'
    
    # create new folder
    num_current_run = num_currnt_run(folder_path)
    new_sub_folder = '{:03d}-{:s}'.format(
        num_current_run, timestamp.strftime("%b%d-%H:%M:%S")
    )
    os.mkdir(os.path.join(folder_path, new_sub_folder))
    os.mkdir(os.path.join(folder_path,new_sub_folder,'plot'))

    return os.path.join(folder_path, new_sub_folder)

def save_optuna_trial(ckpt, timestamp, study_dir, trial_no):
    ckpt_file = '{:03d}-ckpt-{}.pt'.format(trial_no, timestamp.strftime("%Y:%m:%d_%H%M%S"))
    ckpt_path = os.path.join(study_dir, ckpt_file)

    torch.save(ckpt, ckpt_path)

    return ckpt_path

def seed_torch(seed=46):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# -------------- Import functions for data loading --------------
class Dataset_syn(Dataset):
    '''Dataset class for synthetic data
    '''

    def __init__(self, raw_dataset, mix_size = 1):
        super(Dataset_syn, self).__init__()
        self.dataset = raw_dataset
        self.mix_size = mix_size # dimension of mixer output

    def __getitem__(self, index):
        s = self.dataset[index, :self.mix_size]
        x = self.dataset[index, self.mix_size:]

        return x, s

    def __len__(self):
        return len(self.dataset)

def ReadData(dataset_name, train_size=-1, val_size=2000, prob = 1, vline = 0, hline = 0):
    '''Read the data from a csv file. Remove correlation between x_{i0} and Others.
    '''
    file_path = './data/'+dataset_name
    dataset = pd.read_csv(file_path, sep=',')

    # normalization
    dataset = (dataset - dataset.mean())/dataset.std()

    # valset
    valset = dataset.iloc[(len(dataset) - val_size):]

    # training set
    trainset = dataset.iloc[:(len(dataset) - val_size)].sample(train_size, replace = False).reset_index(drop=True)
    ## FOR BIASED DATA: in the case prob < 1, we mask some data of x1 in top right area
    if prob < 1: 
        x1_no_val = dataset.iloc[:(len(dataset) - val_size)][['x1_1', 'x2_1']]
        x1_non_masked_area = x1_no_val[~((x1_no_val['x1_1']>vline)&(x1_no_val['x2_1']>hline))]
        x1_masked_area = x1_no_val[((x1_no_val['x1_1']>vline)&(x1_no_val['x2_1']>hline))]
        x1_masked_area = x1_masked_area.sample(int(prob*len(x1_masked_area)), replace = False).reset_index(drop=True)
        x1_merge = pd.concat((x1_non_masked_area, x1_masked_area), axis=0).reset_index(drop=True)
        x1_trainset = x1_merge.sample(train_size, replace = False).reset_index(drop=True)
        trainset[['x1_1', 'x2_1']] = x1_trainset

    return np.asarray(trainset), np.asarray(valset)

def ReadData_Fix_Cor(dataset_name, train_size=-1, val_size=2000):
    '''Read the data from a csv file. Preserve correlation between x_{i0} and Others.
    '''
    file_path = './data/'+dataset_name
    dataset = pd.read_csv(file_path, sep=',')

    # normalization
    dataset = (dataset - dataset.mean())/dataset.std()

    # valset
    valset = dataset.iloc[(len(dataset) - val_size):]

    # training set
    trainset = dataset.iloc[:(len(dataset) - val_size)].sample(train_size, replace = False).reset_index(drop=True)

    return np.asarray(trainset), np.asarray(valset)
