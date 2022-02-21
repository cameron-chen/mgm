########
# import libraries
########

import argparse
import datetime
import logging
import os

import joblib
import numpy as np
import optuna
import torch
from numpy import random

import eva
import train.train_agent_bias as agentBias
import train.train_agent_low as agentLow
import train.train_mgm_bias as mgmBias
import train.train_mgm_low as mgmLow
import train.train_mixer as mixer
from gan import help

########
# read parameters
########

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=150, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--lambda_gp", type=float, default=10, help="weight of gradient penalty for agent critic")
parser.add_argument("--lrSys", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1Sys", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2Sys", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--lambdaSys", type=float, default=10, help="weight of gradient penalty for system critic in hierarchical setting")
parser.add_argument("--latent_dim", type=int, default=2, help="dimensionality of the latent space")
parser.add_argument("--n_conditions", type=int, default=4, help="number of conditions for sysGAN")
parser.add_argument("--noise_type", type=str, default='gaussian', help="type of the random noise")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--use_gpu", type=int, default=1, help="1: use gpu; 0: not use gpu")
parser.add_argument("--dataset_name", type=str, default='data_roll_b_0.7.csv', help="file name of the dataset")
parser.add_argument("--output_size", type=int, default=2, help="size of generator output")
parser.add_argument("--output_size_agent", type=int, default=2, help="size of agent generator output")
parser.add_argument("--print_interval", type=int, default=10, help="interval of output loss")
parser.add_argument("--data_size", type=int, default=1, help="the size of the data used for experiment")
parser.add_argument("--checkpoint_dir", type=str, default='output/', help="directory of the checkpoint")
parser.add_argument("-ckpt", "--ckpt_name", type=str, default='ckpt.pt', help="name of the checkpoint")
parser.add_argument("--alpha", type=float, default=0, help="weight of the sysGAN loss on agent generator")
parser.add_argument("--seed", type=int, default=-1, help="seed used for control the sampling process of data preprocessing")
parser.add_argument("--trainingMode", type=str, default='optuna', choices=['optuna','singlerun'])
parser.add_argument("--numOptunaTrial", type=int, default=100, help="number of trials on optuna")
parser.add_argument("--optimName", type=str, default='adam', help="Name of the optimizer")
parser.add_argument("--plotOutput", type=str, default='no', help="<agent>: viz output of agent; <sys>: viz output of system; <all>: viz output of both; <no>: default, viz nothing")
parser.add_argument("--trackTraining", type=str, default='no', help="all: print loss and metrics")
parser.add_argument("--object", type = str, default='agent_low', help="training model of interest")
parser.add_argument("--prob", type=float, default=1, help="proportion of data remaining in certain area")
parser.add_argument("--n_record", type=int, default=500, help="time interval to record the model and viz")
parser.add_argument('-note', '--note', type=str, default='-')
parser.add_argument('-rpsl', '--restore_path_lowdataset', type=str, default='', help="folder path of low dataset")
parser.add_argument('-ss', '--save_sample', default=False, action='store_true')


opt = parser.parse_args()
timestamp = help.datetime_as_timezone(datetime.datetime.utcnow())

########
# help function
########

# objective func for optuna
def objective(trial):
    # -------------------- Changeable Chunk --------------------
    # parameter combinations: can be changed
    opt.lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    opt.alpha = trial.suggest_float('alpha', 0.0, 1.0)
    # -------------------- Changeable Chunk --------------------

    # random seed when call objective function
    manualSeed = random.randint(1,10000)
    help.seed_torch(manualSeed)
    trial.set_user_attr('manual seed', manualSeed)

    timestamp_trial = help.datetime_as_timezone(datetime.datetime.utcnow())

    # run models
    if opt.object == 'agent_low':
        checkpoint = agentLow.main(opt, trial = trial, timestamp=timestamp_trial)
    elif opt.object == 'agent_bias':
        checkpoint = agentBias.main(opt, trial = trial, timestamp=timestamp_trial)
    elif opt.object == 'mixer':
        checkpoint = mixer.main(opt, trial = trial, timestamp=timestamp_trial)
    elif opt.object == 'mgm_bias':
        checkpoint = mgmBias.main(opt, trial = trial, timestamp=timestamp_trial)
    elif opt.object == 'mgm_low':
        checkpoint = mgmLow.main(opt, trial = trial, timestamp=timestamp_trial)

    metrics_train = checkpoint['metrics_train']
    final_metric = metrics_train[len(metrics_train)-15: len(metrics_train)-1]
    final_metric = np.mean(final_metric)

    ckpt_path = help.save_optuna_trial(
        checkpoint, timestamp_trial, NEW_STUDY_DIR, trial.number
    )
    trial.set_user_attr('ckpt_path', ckpt_path)

    return final_metric

########
# choose mode of training
########

## optuna
if opt.trainingMode =='optuna':
    ### initialize
    NEW_STUDY_DIR = help.init_optuna_study(opt.object, timestamp)
    help.save_conf(NEW_STUDY_DIR, opt)

    ### logger
    FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
    logPath = os.path.join(NEW_STUDY_DIR, 'train_notes.txt')
    logging.basicConfig(filename=logPath, level=logging.INFO, format=FORMAT)
    logger = logging.getLogger(__name__)

    #### train on GPU or CPU
    cuda = True if opt.use_gpu == 1 and torch.cuda.is_available() else False
    if cuda:
        logger.info('Compute on: GPU\n')
    else:
        logger.info('Compute on: CPU\n')
    logger.info('*NOTE: {}'.format(opt.note))
    logger.info('Given Dataset: {}'.format(
        opt.restore_path_lowdataset if opt.restore_path_lowdataset else 'None'
    ))

    logger.info('Start to run optuna at '+help.datetime_as_timezone(datetime.datetime.utcnow()).strftime("%Y-%m-%d %H:%M:%S"))

    ### train the model
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=8, n_warmup_steps=30, interval_steps=10),
        )
    study.optimize(objective, n_trials=opt.numOptunaTrial)
    logger.info('Complete optuna optimzation at '+help.datetime_as_timezone(datetime.datetime.utcnow()).strftime("%Y-%m-%d %H:%M:%S")+'\n')
    
    ### export coarse results 
    logger.info('## Report ##')
    help.extrResultsOptuna(study, logger) # may be record this in logs

    ### export and select results
    #### export full model
    dfStudy, layoutStudy = help.exportPlotToPdOptuna(study)
    #### select the results: a list containing the recommended trials' names
    listTrialsRec = help.selectBestResultOptuna(dfStudy)

    ### report: report the results in the form of the logs
    filePath = os.path.join(NEW_STUDY_DIR, 'plot','optuna_plot.png')
    figJson = help.reportOptuna(listTrialsRec, dfStudy, layoutStudy,\
        logger,study, fileName = filePath)
    #### Save the plotly figure in the study
    study.set_user_attr('Plot of Top5 Trials', figJson)
    filePath = os.path.join(NEW_STUDY_DIR, 'optuna_study.pkl')
    joblib.dump(study, filePath)
    logger.info("-- Save Study Instance --")
    logger.info('Successfully save the optuna study at: <%s>\n' \
        %(filePath))

    ### evaluate the model
    eva.optuna_eva(listTrialsRec, study, NEW_STUDY_DIR, opt, logger)

    logger.info('Complete the report at '+help.datetime_as_timezone(datetime.datetime.utcnow()).strftime("%Y-%m-%d %H:%M:%S"))
    print('Please find the log file: <%s>' %logPath)

## single run
elif opt.trainingMode == 'singlerun':
    # manual seed
    if opt.seed != -1:
        help.seed_torch(opt.seed)
        print('\nSet the manual seed: %d' %opt.seed)

    # initialize signle run
    opt.log_dir = help.init_singlerun(opt.object, timestamp)
    help.save_conf(opt.log_dir, opt)

    # run models
    if opt.object == 'agent_low':
        checkpoint = agentLow.main(opt, timestamp=timestamp)
    elif opt.object == 'agent_bias':
        checkpoint = agentBias.main(opt, timestamp=timestamp)
    elif opt.object == 'mixer':
        checkpoint = mixer.main(opt, timestamp=timestamp)
    elif opt.object == 'mgm_bias':
        checkpoint = mgmBias.main(opt, timestamp=timestamp)
    elif opt.object == 'mgm_low':
        checkpoint = mgmLow.main(opt, timestamp=timestamp)

    # export the results
    help.plot_loss(checkpoint, opt.log_dir)
    help.save_model(checkpoint, opt.log_dir)

else:
    raise ValueError('Unknown training mode, please select "singlerun" or "optuna".')
