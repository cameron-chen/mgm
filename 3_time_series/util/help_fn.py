import json
import os
import random
import re

import numpy as np
import optuna
import pandas as pd
import plotly
import plotly.io as pio
import torch
from optuna.trial import TrialState
from pytz import timezone
from torch.autograd import Variable


def get_dtypes(use_gpu):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if use_gpu and torch.cuda.is_available():
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype

def get_noise(shape, float_dtype, noise_type='uniform'):
    '''
    generate normal noise or uniform noise

    Paras:
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

def seed_torch(seed=46):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def datetime_as_timezone(date_time, time_zone = 'Asia/Singapore'):
    """get Singapore DateTime
    """
    tz = timezone(time_zone)
    utc = timezone('UTC')
    return date_time.replace(tzinfo=utc).astimezone(tz)

# ------------ help func: optuna ------------

def extrResultsOptuna(study, save_file):
    '''
    extract the useful information of Optuna model (named study)
    '''
    with open("./output/{}/train_notes.txt".format(save_file), 'a') as f:
        f.write("-- Study statistics --\n")
        f.write("Number of finished trials: %d\n" %len(study.trials))
        f.write("Number of pruned trials: %d\n" %len(study.get_trials(deepcopy=False, states=[TrialState.PRUNED])))
        f.write("Number of complete trials: %d\n\n" %len(study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])))

        f.write("-- Best trial --\n")
        trial = study.best_trial

        f.write("Value (min metric): %.5f\n" %trial.value)
        f.write("Trial Num: %d\n" %trial.number)

        try:
            trial.user_attrs['manual seed']
        except:
            f.write("Seed: no seed recorded in this study instance\n")
        else:
            f.write("Seed: %d\n" %trial.user_attrs['manual seed'])

        f.write("Params: \n")
        for key, value in trial.params.items():
            f.write("  {}: {}\n".format(key, value))

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
    '''Return names of the top 5 trials
    '''
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

    return trialsBest

def plotSelectResult(trails_best, df, plotly_layout, save_file):
    df_top5_trials = df[df['name'].isin(trails_best)]
    # plot plotly
    fig = plotOptuna(df_top5_trials, plotly_layout, trails_best)
    # export plotly figure to png
    exportPlotOptuna(fig, save_file)

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

def exportPlotOptuna(figure, save_file):
    filePath = './output/{}/plot/top5_trials.png'.format(save_file)
    pio.write_image(figure, filePath)

    # log: successfully write the top 5 trials image
    with open("./output/{}/train_notes.txt".format(save_file), 'a') as f:
        f.write('\n-- Save Plot --\n')
        f.write('Top 5 trials (png form) have been saved in: <%s>\n' %filePath)
        # log: record the dir of optuna (in study) and png
        f.write('Top 5 trials (Plotly object, Json form) would be stored in optuna.study.user_attrs\n'+\
            'Please use <Plotly.io.from_json> to recover\n')

def reportOptuna(trails_best, df, save_file, study):

    with open("./output/{}/train_notes.txt".format(save_file), 'a') as f:
        f.write('\n-- Trial Details --\n')
        for trialName in trails_best:
            dfTrial = df[df['name']==trialName]
            seriesTop10 = [i for i in dfTrial['y'].nsmallest(10)]
            trialNum = int(re.search(r'\d+', trialName).group(0))
            Trial = study.trials[trialNum]

            f.write('The trial ID is: %s\n' %trialName)
            f.write('The 1st smallest metric value is %.3f\n'%seriesTop10[0]) 
            f.write('The 3rd smallest metric value is %.3f\n'%seriesTop10[2])
            f.write('The 10th smallest metric value is %.3f\n'%seriesTop10[9])
            f.write('The top 10 smallest metric values are: ['+\
                ', '.join(str(round(e,3)) for e in seriesTop10)+']\n')
            f.write('Trial details:\n')
            f.write("  Avg value (avg of top min 6): %.5f\n" %Trial.value)
            try:
                Trial.user_attrs['manual seed']
            except:
                f.write("  Seed: no seed recorded in this study instance\n")
            else:
                f.write("  Seed: %d\n" %Trial.user_attrs['manual seed'])
            try:
                Trial.user_attrs['save_file']
            except:
                f.write("  Trained file: no trained file in this study instance\n")
            else:
                f.write("  Trained file: {}\n".format(Trial.user_attrs['save_file']))
            f.write("  Params: \n")
            for key, value in Trial.params.items():
                f.write("    {}: {}\n".format(key, value))
            f.write('\n')
