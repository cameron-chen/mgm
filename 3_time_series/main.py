#
# import libraries
#
import argparse
import datetime
import os
import random

import joblib
import numpy as np
import optuna
import torch
from optuna import study

import cot_base as cot_base
import cot_condition as cot_cond
import cot_condition_mod as cot_rec
import cot_hier as cot_hier
from util import help_fn

#
# read parameters
#
parser = argparse.ArgumentParser(description='cot')
parser.add_argument('-d', '--dname', type=str, default='AROne',
                        choices=['AROne', 'Elec', 'Elec_low'])
parser.add_argument('-s', '--seed', type=int, default=42)
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
parser.add_argument('-rs', '--reg_penalty_sys', type=float, default=10.0)
parser.add_argument('-sinkes', '--sinkhorn_eps_sys', type=float, default=100)
parser.add_argument('-rl', '--rec_lambda', type=float, default=1)
parser.add_argument('-a', '--alpha', type=float, default=0)
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
parser.add_argument('-lrs', '--lr_sys', type=float, default=1e-3)
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
parser.add_argument('-rp', '--restore_path', type=str, default="")
parser.add_argument('-rpl', '--restore_path_lowdataset', type=str, default="")
parser.add_argument('-log', '--tb_log', type=int, default=0,
                    choices=[0, 1], help='tensorboard logs')
parser.add_argument('-tm', '--training_mode', type=str, default='single_run',
                    choices=['optuna','single_run'])
parser.add_argument('-obj', '--object', type=str, default='agent',
                    help='goal model to optimize')
parser.add_argument('-not', '--num_optuna_trial', type=int, default=10)
parser.add_argument('-pruner', '--pruner', type=int, default=0,
                    choices=[0, 1])
parser.add_argument('-rc', '--reconstruction', type=int, default=0,
                    choices=[0, 1])
parser.add_argument('-sn', '--study_name', type=str, default='')
parser.add_argument('-note', '--note', type=str, default='-')

opt = parser.parse_args()
timestamp = help_fn.datetime_as_timezone(datetime.datetime.utcnow())

# training notes
if opt.training_mode in ['optuna']:
    cuda = True if bool(opt.use_gpu) and torch.cuda.is_available() else False
    _test = opt.dname+'_main'
    save_file = "{}_{}".format(_test, timestamp.strftime("%h%d-%H:%M:%S.%f"))

    if not os.path.exists('./output/{}/plot'.format(save_file)):
        os.makedirs('./output/{}/plot'.format(save_file))
    
    # GAN train notes
    with open("./output/{}/train_notes.txt".format(save_file), 'w') as f:
        # Include any experiment notes here:
        f.write("Experiment notes: .... \n\n")
        f.write("*Note: \n{}\n\n".format(opt.note))
        if cuda:
            f.write('Compute on: GPU\n\n')
        else:
            f.write('Compute on: CPU\n\n')
        f.write('Training start at {}\n'.format(
            help_fn.datetime_as_timezone(datetime.datetime.utcnow()).strftime("%Y-%m-%d %H:%M:%S")))

#
# help function
#
def objective(trial):
    '''Objective function of optuna
    '''
    # -------------------- Changeable Chunk --------------------
    opt.lr = trial.suggest_float('lr', 5e-5, 5e-4, log=True)
    opt.lr_sys = trial.suggest_float('lr mixer', 1e-5, 5e-4, log=True)
    opt.alpha = trial.suggest_float('alpha', 0.35, 1.0, step=0.05)
    # -------------------- Changeable Chunk --------------------

    # random seed when call objective function
    manualSeed = random.randint(1,10000)
    help_fn.seed_torch(manualSeed)
    trial.set_user_attr('manual seed', manualSeed)

    timestamp_trial = help_fn.datetime_as_timezone(datetime.datetime.utcnow())

    # run models
    if opt.object == 'agent':
        checkpoint = cot_base.main(opt, trial = trial, timestamp=timestamp_trial)
    elif opt.object == 'mixer':
        checkpoint = cot_cond.main(opt, trial = trial, timestamp=timestamp_trial)
    elif opt.object == 'hier':
        checkpoint = cot_hier.main(opt, trial = trial, timestamp=timestamp_trial)
    elif opt.object == 'mixer_rec':
        checkpoint = cot_rec.main(opt, trial = trial, timestamp=timestamp_trial)

    metrics = checkpoint['metric']
    top_6 = np.sort(metrics)[:6]

    return np.mean(top_6)

#
# training mode
# 
if opt.training_mode == 'optuna':
    study = optuna.load_study(
        study_name=opt.study_name,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=8, n_warmup_steps=60, interval_steps=10),
    )
    study.optimize(objective, n_trials=opt.num_optuna_trial)

    with open("./output/{}/train_notes.txt".format(save_file), 'a') as f:
        f.write('Complete training at {}\n\n'.format(
            help_fn.datetime_as_timezone(datetime.datetime.utcnow()).strftime("%Y-%m-%d %H:%M:%S")))

    # record overall results
    help_fn.extrResultsOptuna(study, save_file)

    # record detailed results
    ## export and select results
    df_study, layout_study = help_fn.exportPlotToPdOptuna(study)
    selected_trails = help_fn.selectBestResultOptuna(df_study)

    ## plot the selected results
    figJson = help_fn.plotSelectResult(selected_trails, df_study, layout_study, save_file)
    study.set_user_attr('Plot of Top5 Trials', figJson)

    ## record the selected results
    help_fn.reportOptuna(selected_trails, df_study, save_file, study)

    # save optuna file /study instance
    joblib.dump(study, "./output/{}/optuna_instance.pkl".format(save_file))
    with open("./output/{}/train_notes.txt".format(save_file), 'a') as f:
        f.write("\n-- Save Study Instance --\n")
        f.write('Successfully save the optuna study at: <%s>\n' \
            %("./output/{}/optuna_instance.pkl".format(save_file)))
        f.write('Complete the report at '+
                help_fn.datetime_as_timezone(datetime.datetime.utcnow()).strftime("%Y-%m-%d %H:%M:%S")+
                '\n')
    print('Please find the log file: <./output/{}/train_notes.txt>'.format(save_file))

elif opt.training_mode == 'single_run':
    # random seed when call objective function
    manualSeed = random.randint(1,10000)
    help_fn.seed_torch(manualSeed)

    timestamp = help_fn.datetime_as_timezone(datetime.datetime.utcnow())

    # run models
    if opt.object == 'agent':
        checkpoint = cot_base.main(opt, timestamp=timestamp)
    elif opt.object == 'mixer':
        checkpoint = cot_cond.main(opt, timestamp=timestamp)
    elif opt.object == 'hier':
        checkpoint = cot_hier.main(opt, timestamp=timestamp)
    elif opt.object == 'mixer_rec':
        checkpoint = cot_rec.main(opt, timestamp=timestamp)
else:
    ValueError('Please specify training mode: optuna or single_run')
