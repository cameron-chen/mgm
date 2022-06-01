# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Main training script."""
import os
from unittest.mock import patch
import requests
import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary
import PIL.Image
import zipfile

from training import dataset
from training import misc
from metrics import metric_base
from collections import OrderedDict

#----------------------------------------------------------------------------
# Just-in-time processing of training images before feeding them to the networks.

def process_reals(x, labels, lod, mirror_augment, drange_data, drange_net):
    """
    Args:
     - mirror_augment: mirror image
     - drange_data: [0, 255] in original implementation
     - drange_net: [-1, 1] in original implementation, it uses MinMax scaling
    """
    with tf.name_scope('DynamicRange'):
        x = tf.cast(x, tf.float32)
        x = misc.adjust_dynamic_range(x, drange_data, drange_net)
    if mirror_augment:
        with tf.name_scope('MirrorAugment'):
            x = tf.where(tf.random_uniform([tf.shape(x)[0]]) < 0.5, x, tf.reverse(x, [3]))
    with tf.name_scope('FadeLOD'): # Smooth crossfade between consecutive levels-of-detail.
        s = tf.shape(x)
        y = tf.reshape(x, [-1, s[1], s[2]//2, 2, s[3]//2, 2])
        y = tf.reduce_mean(y, axis=[3, 5], keepdims=True)
        y = tf.tile(y, [1, 1, 1, 2, 1, 2])
        y = tf.reshape(y, [-1, s[1], s[2], s[3]])
        x = tflib.lerp(x, y, lod - tf.floor(lod))
    with tf.name_scope('UpscaleLOD'): # Upscale to match the expected input/output size of the networks.
        s = tf.shape(x)
        factor = tf.cast(2 ** tf.floor(lod), tf.int32)
        x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
        x = tf.tile(x, [1, 1, 1, factor, 1, factor])
        x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
    return x, labels

#----------------------------------------------------------------------------
# Evaluate time-varying training parameters.

def training_schedule(
    cur_nimg,
    training_set,
    lod_initial_resolution  = None,     # Image resolution used at the beginning.
    lod_training_kimg       = 600,      # Thousands of real images to show before doubling the resolution.
    lod_transition_kimg     = 600,      # Thousands of real images to show when fading in new layers.
    minibatch_size_base     = 32,       # Global minibatch size.
    minibatch_size_dict     = {},       # Resolution-specific overrides.
    minibatch_gpu_base      = 4,        # Number of samples processed at a time by one GPU.
    minibatch_gpu_dict      = {},       # Resolution-specific overrides.
    G_lrate_base            = 0.002,    # Learning rate for the generator.
    G_lrate_dict            = {},       # Resolution-specific overrides.
    D_lrate_base            = 0.002,    # Learning rate for the discriminator.
    D_lrate_dict            = {},       # Resolution-specific overrides.
    lrate_rampup_kimg       = 0,        # Duration of learning rate ramp-up.
    tick_kimg_base          = 4,        # Default interval of progress snapshots.
    tick_kimg_dict          = {8:28, 16:24, 32:20, 64:16, 128:12, 256:8, 512:6, 1024:4}): # Resolution-specific overrides.

    # Initialize result dict.
    s = dnnlib.EasyDict()
    s.kimg = cur_nimg / 1000.0

    # Training phase.
    phase_dur = lod_training_kimg + lod_transition_kimg
    phase_idx = int(np.floor(s.kimg / phase_dur)) if phase_dur > 0 else 0
    phase_kimg = s.kimg - phase_idx * phase_dur

    # Level-of-detail and resolution.
    if lod_initial_resolution is None:
        s.lod = 0.0
    else:
        s.lod = training_set.resolution_log2
        s.lod -= np.floor(np.log2(lod_initial_resolution))
        s.lod -= phase_idx
        if lod_transition_kimg > 0:
            s.lod -= max(phase_kimg - lod_training_kimg, 0.0) / lod_transition_kimg
        s.lod = max(s.lod, 0.0)
    s.resolution = 2 ** (training_set.resolution_log2 - int(np.floor(s.lod)))

    # Minibatch size.
    s.minibatch_size = minibatch_size_dict.get(s.resolution, minibatch_size_base)
    s.minibatch_gpu = minibatch_gpu_dict.get(s.resolution, minibatch_gpu_base)

    # Learning rate.
    s.G_lrate = G_lrate_dict.get(s.resolution, G_lrate_base)
    s.D_lrate = D_lrate_dict.get(s.resolution, D_lrate_base)
    if lrate_rampup_kimg > 0:
        rampup = min(s.kimg / lrate_rampup_kimg, 1.0)
        s.G_lrate *= rampup
        s.D_lrate *= rampup

    # Other parameters.
    s.tick_kimg = tick_kimg_dict.get(s.resolution, tick_kimg_base)
    return s

def print_g_def(graph_def, output_path):
    with open(output_path, 'w') as f:
        for n in graph_def.node:
            f.write(n.name+'\n')
    
    print('write the graph structure at "{}"'.format(output_path))

def print_gradient_stat(grads_vars_pair, vars):
    print('Mean absolute gradients: ')
    for i in range(len(vars)):
        _grad = grads_vars_pair[i][0]
        w_stat = np.abs(_grad).mean()
        print('{}._grad: {}'.format(vars[i].name, w_stat))

def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination) 

def download_pretrain_model():
    import zipfile

    file_path_online = '1RVIBe_h6ttvVTPslT-MvAz-4U76iY6kP'
    destination_path = 'checkpoint/system_models/cyclegan.zip'
    download_file_from_google_drive(file_path_online, destination_path)

    with zipfile.ZipFile(destination_path, 'r') as zip_ref:
        zip_ref.extractall('checkpoint/system_models')
    
    # verify the code
    if not os.path.exists('checkpoint/system_models/cyclegan'):
        AssertionError("The pretrained CycleGAN does not download properly, please donwload it from <{}> and extract it in <{}>".format(
            'https://drive.google.com/file/d/1RVIBe_h6ttvVTPslT-MvAz-4U76iY6kP/view?usp=sharing',
            './checkpoint/system_models'
        ))

#----------------------------------------------------------------------------
# Main training script.

def training_loop(
    G_args                  = {},       # Options for generator network.
    D_args                  = {},       # Options for discriminator network.
    G_opt_args              = {},       # Options for generator optimizer.
    D_opt_args              = {},       # Options for discriminator optimizer.
    AE_opt_args             = None,      # Options for autoencoder optimizer.
    G_loss_args             = {},       # Options for generator loss.
    D_loss_args             = {},       # Options for discriminator loss.
    AE_loss_args            = None,      # Options for autoencoder loss.
    dataset_args            = {},       # Options for dataset.load_dataset().
    dataset_args_eval       = {},       # Options for dataset.load_dataset().
    sched_args              = {},       # Options for train.TrainingSchedule.
    grid_args               = {},       # Options for train.setup_snapshot_image_grid().
    metric_arg_list         = [],       # Options for MetricGroup.
    tf_config               = {},       # Options for tflib.init_tf().
    train_data_dir          = None,     # Directory to load datasets from.
    eval_data_dir           = None,     # Directory to load datasets from.
    G_smoothing_kimg        = 10.0,     # Half-life of the running average of generator weights.
    minibatch_repeats       = 4,        # Number of minibatches to run before adjusting training parameters.
    lazy_regularization     = True,     # Perform regularization as a separate training step?
    G_reg_interval          = 4,        # How often the perform regularization for G? Ignored if lazy_regularization=False.
    D_reg_interval          = 16,       # How often the perform regularization for D? Ignored if lazy_regularization=False.
    reset_opt_for_new_lod   = True,     # Reset optimizer internal state (e.g. Adam moments) when new layers are introduced?
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    mirror_augment          = False,    # Enable mirror augment?
    drange_net              = [-1,1],   # Dynamic range used when feeding image data to the networks.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = only save 'reals.png' and 'fakes-init.png'.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = only save 'networks-final.pkl'.
    save_tf_graph           = True,    # Include full TensorFlow computation graph in the tfevents file?
    save_weight_histograms  = True,    # Include weight histograms in the tfevents file?
    resume_pkl              = None,     # Network pickle to resume training from, None = train from scratch.
    resume_kimg             = 0.0,      # Assumed training progress at the beginning. Affects reporting and training schedule.
    resume_time             = 0.0,      # Assumed wallclock time at the beginning. Affects reporting.
    hier_training           = False,    # train a hierarchcial model
    SYS_args                = {},       # Options for system
    resume_with_new_nets    = False,
    resume_with_own_vars    = False):   # Construct new networks according to G_args and D_args before resuming training?


    # Initialize dnnlib and TensorFlow.
    tflib.init_tf(tf_config)
    num_gpus = dnnlib.submit_config.num_gpus

    # Load training set.
    print("Loading train set from %s..." % dataset_args.tfrecord_dir)
    training_set = dataset.load_dataset(data_dir=dnnlib.convert_path(train_data_dir), verbose=True, **dataset_args)
    print("Loading eval set from %s..." % dataset_args_eval.tfrecord_dir)
    eval_set = dataset.load_dataset(data_dir=dnnlib.convert_path(eval_data_dir), verbose=True, **dataset_args_eval)
    grid_size, grid_reals, grid_labels = misc.setup_snapshot_image_grid(training_set, **grid_args)
    misc.save_image_grid(grid_reals, dnnlib.make_run_dir_path('reals.png'), drange=training_set.dynamic_range, grid_size=grid_size)
    ## Load extra training data in hierarchcial model
    if hier_training:
        def scaling_function(img_tensor):
            return (img_tensor/127.5 - 1)
        train_data_sys_dir = './data/zebra'
        print("Loading extra train set from '{}' for hierarchical training"\
            .format(train_data_sys_dir))
        idg = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, preprocessing_function=scaling_function)
        training_set_sys = tf.keras.preprocessing.image.DirectoryIterator(
            train_data_sys_dir, idg, data_format='channels_last',
            batch_size=4, class_mode=None
        ) #NOTE: this dataset output (?, 256, 256, 3) with element value in [-1, 1]

    # Freeze Discriminator
    if D_args['freeze']:
        num_layers = np.log2(training_set.resolution) - 1
        layers = int(np.round(num_layers * 3. / 8.))
        scope = ['Output', 'scores_out']
        for layer in range(layers):
            scope += ['.*%d' % 2**layer]
            if 'train_scope' in D_args: scope[-1] += '.*%d' % D_args['train_scope']
        D_args['train_scope'] = scope

    # Update certain layers via L_sys
    # NOTE: SYS_args['freeze_G_sysloss']    # freeze layers of G_a optimized by L_sys starting at <freeze_G_sysloss>. [1, 7]
    # NOTE: SYS_args['freeze_D']            # freeze first <freeze_D> layers of discriminator. [1, 7]
    if hier_training:
        SYS_args['untrainable'] = list()    # untrainable layers of G_a via L_sys
        if SYS_args['freeze_G_sysloss']:
            for layer in range(SYS_args['freeze_G_sysloss'], 8):
                _res = 2**(layer+1)
                SYS_args['untrainable'].append('{}x{}'.format(_res, _res))
            print("Freeze layer <{}> of G_a when use L_sys".format(', '.join(SYS_args['untrainable'])))
    if SYS_args['freeze_D']:
        # help func, del key in a dictionary
        def removekey(dic, key):
            r = OrderedDict(dic)
            del r[key]
            return r
        # inspect the trainable scope
        D_args['untrainable'] = list()
        freeze_D_ = 8-SYS_args['freeze_D']
        for layer in range(freeze_D_,8):
            _res = 2**(layer+1)
            D_args['untrainable'].append('{}x{}'.format(_res,_res))
        print('Freeze layer <{}> of D_a'.format(', '.join(D_args['untrainable'])))
    # Construct or load networks.
    with tf.device('/gpu:0'):
        if resume_pkl is '' or resume_with_new_nets or resume_with_own_vars:
            print('Constructing networks...')
            G = tflib.Network('G', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=training_set.label_size, **G_args)
            D = tflib.Network('D', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=training_set.label_size, **D_args)
            Gs = G.clone('Gs')
        if resume_pkl is not '':
            print('Loading networks from "%s"...' % resume_pkl)
            rG, rD, rGs = misc.load_pkl(resume_pkl)
            if resume_with_new_nets: 
                G.copy_vars_from(rG); 
                D.copy_vars_from(rD); 
                Gs.copy_vars_from(rGs)
            else: G = rG; D = rD; Gs = rGs

    grid_latents = np.random.randn(np.prod(grid_size), *G.input_shape[1:])
    # SVD stuff
    if 'syn_svd' in G_args or 'map_svd' in G_args:
        # Run graph to calculate SVD
        grid_latents_smol = grid_latents[:1] 
        rho = np.array([1])
        grid_fakes = G.run(grid_latents_smol, grid_labels, rho, is_validation=True)
        grid_fakes = Gs.run(grid_latents_smol, grid_labels, rho, is_validation=True)
        load_d_fake = D.run(grid_reals[:1], rho, is_validation=True)
        with tf.device('/gpu:0'):
            # Create SVD-decomposed graph
            rG, rD, rGs = G, D, Gs
            G_lambda_mask = {var: np.ones(G.vars[var].shape[-1]) for var in G.vars if 'SVD/s' in var}
            D_lambda_mask = {'D/' + var: np.ones(D.vars[var].shape[-1]) for var in D.vars if 'SVD/s' in var}
            G_reduce_dims = {var: (0, int(Gs.vars[var].shape[-1])) for var in Gs.vars if 'SVD/s' in var}
            G_args['lambda_mask'] = G_lambda_mask
            G_args['reduce_dims'] = G_reduce_dims
            D_args['lambda_mask'] = D_lambda_mask

            # Create graph with no SVD operations
            G = tflib.Network('G', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=rG.input_shapes[1][1], factorized=True, **G_args)
            D = tflib.Network('D', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=rD.input_shapes[1][1], factorized=True, **D_args)
            Gs = G.clone('Gs')

            grid_fakes = G.run(grid_latents_smol, grid_labels, rho, is_validation=True, minibatch_size=1)
            grid_fakes = Gs.run(grid_latents_smol, grid_labels, rho, is_validation=True, minibatch_size=1)

            G.copy_vars_from(rG)
            D.copy_vars_from(rD)
            Gs.copy_vars_from(rGs)

    # Reduce per-gpu minibatch size to fit in 16GB GPU memory
    if grid_reals.shape[2] >= 1024:
        sched_args.minibatch_gpu_base = 2
    print('Batch size', sched_args.minibatch_gpu_base)

    # Generate initial image snapshot.
    G.print_layers(); D.print_layers()
    sched = training_schedule(cur_nimg=total_kimg*1000, training_set=training_set, **sched_args)
    grid_latents = np.random.randn(np.prod(grid_size), *G.input_shape[1:])
    rho = np.array([1])
    grid_fakes = Gs.run(grid_latents, grid_labels, rho, is_validation=True, minibatch_size=sched.minibatch_gpu)
    misc.save_image_grid(grid_fakes, dnnlib.make_run_dir_path('fakes_init.png'), drange=drange_net, grid_size=grid_size)
    if resume_pkl is not '':
        load_d_real = rD.run(grid_reals[:1], rho, is_validation=True)
        load_d_fake = rD.run(grid_fakes[:1], rho, is_validation=True)
        d_fake = D.run(grid_fakes[:1], rho, is_validation=True)
        d_real = D.run(grid_reals[:1], rho, is_validation=True)
        print('Factorized fake', d_fake, 'loaded fake', load_d_fake, 'factorized real', d_real, 'loaded real', load_d_real)
        print('(should match)')
    # Setup training inputs.
    print('Building TensorFlow graph...')
    with tf.name_scope('Inputs'), tf.device('/cpu:0'):
        lod_in               = tf.placeholder(tf.float32, name='lod_in', shape=[])
        lrate_in             = tf.placeholder(tf.float32, name='lrate_in', shape=[])
        minibatch_size_in    = tf.placeholder(tf.int32, name='minibatch_size_in', shape=[])
        minibatch_gpu_in     = tf.placeholder(tf.int32, name='minibatch_gpu_in', shape=[])
        minibatch_multiplier = minibatch_size_in // (minibatch_gpu_in * num_gpus)
        Gs_beta              = 0.5 ** tf.div(tf.cast(minibatch_size_in, tf.float32), G_smoothing_kimg * 1000.0) if G_smoothing_kimg > 0.0 else 0.0

    # Setup optimizers.
    G_opt_args = dict(G_opt_args)
    D_opt_args = dict(D_opt_args)
    for args, reg_interval in [(G_opt_args, G_reg_interval), (D_opt_args, D_reg_interval)]:
        args['minibatch_multiplier'] = minibatch_multiplier
        args['learning_rate'] = lrate_in
        if lazy_regularization:
            mb_ratio = reg_interval / (reg_interval + 1)
            args['learning_rate'] *= mb_ratio
            if 'beta1' in args: args['beta1'] **= mb_ratio
            if 'beta2' in args: args['beta2'] **= mb_ratio
    G_opt = tflib.Optimizer(name='TrainG', **G_opt_args)
    D_opt = tflib.Optimizer(name='TrainD', **D_opt_args)
    G_reg_opt = tflib.Optimizer(name='RegG', share=G_opt, **G_opt_args)
    D_reg_opt = tflib.Optimizer(name='RegD', share=D_opt, **D_opt_args)
    if AE_opt_args is not None:
        AE_opt_args = dict(AE_opt_args)
        AE_opt_args['minibatch_multiplier'] = minibatch_multiplier
        AE_opt_args['learning_rate'] = lrate_in
        AE_opt = tflib.Optimizer(name='TrainAE', **AE_opt_args)


    # Build training graph for each GPU.
    data_fetch_ops = []
    for gpu in range(num_gpus):
        with tf.name_scope('GPU%d' % gpu), tf.device('/gpu:%d' % gpu):

            # Create GPU-specific shadow copies of G and D.
            G_gpu = G if gpu == 0 else G.clone(G.name + '_shadow')
            D_gpu = D if gpu == 0 else D.clone(D.name + '_shadow')

            # Fetch training data via temporary variables.
            with tf.name_scope('DataFetch'): # NOTE: author sidestep setting placeholder via temp variables
                sched = training_schedule(cur_nimg=int(resume_kimg*1000), training_set=training_set, **sched_args)
                reals_var = tf.Variable(name='reals', trainable=False, initial_value=tf.zeros([sched.minibatch_gpu] + training_set.shape))
                labels_var = tf.Variable(name='labels', trainable=False, initial_value=tf.zeros([sched.minibatch_gpu, training_set.label_size]))
                reals_write, labels_write = training_set.get_minibatch_tf()
                reals_write, labels_write = process_reals(reals_write, labels_write, lod_in, mirror_augment, training_set.dynamic_range, drange_net)
                reals_write = tf.concat([reals_write, reals_var[minibatch_gpu_in:]], axis=0)
                labels_write = tf.concat([labels_write, labels_var[minibatch_gpu_in:]], axis=0)
                data_fetch_ops += [tf.assign(reals_var, reals_write)]
                data_fetch_ops += [tf.assign(labels_var, labels_write)]
                reals_read = reals_var[:minibatch_gpu_in]
                labels_read = labels_var[:minibatch_gpu_in]

            # Evaluate loss functions.
            lod_assign_ops = []
            if 'lod' in G_gpu.vars: lod_assign_ops += [tf.assign(G_gpu.vars['lod'], lod_in)]
            if 'lod' in D_gpu.vars: lod_assign_ops += [tf.assign(D_gpu.vars['lod'], lod_in)]
            with tf.control_dependencies(lod_assign_ops):
                with tf.name_scope('G_loss'):
                    if G_loss_args['func_name'] == 'training.loss.G_l1': G_loss_args['reals'] = reals_read
                    else: G_loss, G_reg = dnnlib.util.call_func_by_name(G=G_gpu, D=D_gpu, opt=G_opt, training_set=training_set, minibatch_size=minibatch_gpu_in, **G_loss_args)
                with tf.name_scope('D_loss'):
                    D_loss, D_reg = dnnlib.util.call_func_by_name(G=G_gpu, D=D_gpu, opt=D_opt, training_set=training_set, minibatch_size=minibatch_gpu_in, reals=reals_read, labels=labels_read, **D_loss_args)

            # Register gradients.
            if SYS_args['freeze_D']:
                for _key in D_gpu.trainables.keys():
                    if any(fil in _key for fil in D_args['untrainable']):
                        D_gpu.trainables = removekey(D_gpu.trainables, _key)

            if not lazy_regularization:
                if G_reg is not None: G_loss += G_reg
                if D_reg is not None: D_loss += D_reg
            else:
                if G_reg is not None: G_reg_opt.register_gradients(tf.reduce_mean(G_reg * G_reg_interval), G_gpu.trainables)
                if D_reg is not None: D_reg_opt.register_gradients(tf.reduce_mean(D_reg * D_reg_interval), D_gpu.trainables)


            G_opt.register_gradients(tf.reduce_mean(G_loss), G_gpu.trainables)
            D_opt.register_gradients(tf.reduce_mean(D_loss), D_gpu.trainables)

    # load system
    if hier_training:
        if not os.path.exists("checkpoint/system_models"):
            os.mkdir('checkpoint/system_models')
        if not os.path.exists("checkpoint/system_models/cyclegan.zip"):
            download_pretrain_model()
        if not os.path.exists("checkpoint/system_models/cyclegan"):
            with zipfile.ZipFile("checkpoint/system_models/cyclegan.zip", 'r') as zip_ref:
                zip_ref.extractall('checkpoint/system_models')

        print("Loading system networks from '{}'...".format("checkpoint/system_models/cyclegan"))
        (real_B_sys,
        output_G_sys,
        DB_fake,
        fake_B_sys,
        D_loss_sys
        )\
        = misc.load_ckpt(
            tf.get_default_session(), "checkpoint/system_models/cyclegan", 
            G, minibatch_gpu_in
        )
        lrate_in_sys = tf.get_default_graph().get_tensor_by_name('cyclegan/learning_rate:0')
        print("Retrieve key nodes of the system: ")
        print([var.name for var in tf.get_collection('key_node')]+[lrate_in_sys.name])
    # system loss
    if hier_training:
        def mae_criterion(in_, target):
            return tf.reduce_mean((in_-target)**2)
        with tf.name_scope('sys_loss'):
            G_loss_sys = mae_criterion(DB_fake, tf.ones_like(DB_fake))
        #NOTE: D_loss_sys has already defined in the loaded model
        # optimizer of D_sys and G_a
        t_vars = tf.trainable_variables()
        d_sys_vars = [var for var in t_vars if 'cyclegan/discriminatorB' in var.name]
        assert isinstance(G.trainables, dict)
        g_vars_sys = [var for var in G.trainables.values() if all(fil not in var.name for fil in SYS_args['untrainable'])]
        with tf.name_scope('System_op'): #NOTE: name them in system operation scope to avoid duplicated name
            D_opt_sys = tf.train.AdamOptimizer(lrate_in_sys, beta1=0.5, name='Adam_sys')\
                .minimize(D_loss_sys, var_list=d_sys_vars)
            G_opt_from_sys = tf.train.AdamOptimizer(lrate_in_sys, beta1=0.5, name='Adam_sys')\
                .minimize(G_loss_sys, var_list=g_vars_sys)


        #NOTE: test code, extract gradient
        # _opt_a = tf.train.AdamOptimizer(learning_rate=1, name='Adam_agent')
        # _opt_s = tf.train.AdamOptimizer(learning_rate=1, name='Adam_system')
        # _gra_a = _opt_a.compute_gradients(G_loss, g_vars_sys)
        # _gra_s = _opt_s.compute_gradients(G_loss_sys, g_vars_sys)

    # Setup training ops.
    data_fetch_op = tf.group(*data_fetch_ops)
    G_train_op = G_opt.apply_updates()
    D_train_op = D_opt.apply_updates()
    G_reg_op = G_reg_opt.apply_updates(allow_no_op=True)
    D_reg_op = D_reg_opt.apply_updates(allow_no_op=True)
    Gs_update_op = Gs.setup_as_moving_average_of(G, beta=Gs_beta)

    # Finalize graph.
    with tf.device('/gpu:0'):
        try:
            peak_gpu_mem_op = tf.contrib.memory_stats.MaxBytesInUse()
        except tf.errors.NotFoundError:
            peak_gpu_mem_op = tf.constant(0)
    tflib.init_uninitialized_vars()

    print('Initializing logs...')
    summary_log = tf.summary.FileWriter(dnnlib.make_run_dir_path())
    if save_tf_graph:
        summary_log.add_graph(tf.get_default_graph())
    if save_weight_histograms:
        G.setup_weight_histograms(); D.setup_weight_histograms()
    metrics = metric_base.MetricGroup(metric_arg_list)

    print('Training for %d kimg...\n' % total_kimg)
    dnnlib.RunContext.get().update('', cur_epoch=resume_kimg, max_epoch=total_kimg)
    maintenance_time = dnnlib.RunContext.get().get_last_update_interval()
    cur_nimg = int(resume_kimg * 1000)
    cur_tick = -1
    tick_start_nimg = cur_nimg
    prev_lod = -1.0
    running_mb_counter = 0

    while cur_nimg < total_kimg * 1000:
        if dnnlib.RunContext.get().should_stop(): break

        # Choose training parameters and configure training ops.
        sched = training_schedule(cur_nimg=cur_nimg, training_set=training_set, **sched_args)
        assert sched.minibatch_size % (sched.minibatch_gpu * num_gpus) == 0
        training_set.configure(sched.minibatch_gpu, sched.lod)
        if reset_opt_for_new_lod:
            if np.floor(sched.lod) != np.floor(prev_lod) or np.ceil(sched.lod) != np.ceil(prev_lod):
                G_opt.reset_optimizer_state(); D_opt.reset_optimizer_state()
        prev_lod = sched.lod

        # Run training ops
        feed_dict = {lod_in: sched.lod, lrate_in: sched.G_lrate, minibatch_size_in: sched.minibatch_size, minibatch_gpu_in: sched.minibatch_gpu}
        for _repeat in range(minibatch_repeats):
            rounds = range(0, sched.minibatch_size, sched.minibatch_gpu * num_gpus)
            ae_iter_mul = 10
            ae_rounds = range(0, sched.minibatch_size, sched.minibatch_gpu * num_gpus * ae_iter_mul)
            run_G_reg = (lazy_regularization and running_mb_counter % G_reg_interval == 0)
            run_D_reg = (lazy_regularization and running_mb_counter % D_reg_interval == 0)
            cur_nimg += sched.minibatch_size
            running_mb_counter += 1

            # Fast path without gradient accumulation.
            if len(rounds) == 1:
                tflib.run([G_train_op, data_fetch_op], feed_dict)
                if run_G_reg:
                    tflib.run(G_reg_op, feed_dict)
                # update G via system loss
                if hier_training:
                    # load minibatch
                    real_B_sys_img = training_set_sys.next()
                    while real_B_sys_img.shape[0] != sched.minibatch_gpu:
                        real_B_sys_img = training_set_sys.next()
                    #NOTE: shape (?, 256, 256, 6) [-1, 1]
                    # session run
                    fake_B_sys_img, _ = tf.get_default_session().run(
                        [output_G_sys, G_opt_from_sys],
                        feed_dict={lrate_in_sys: sched.G_lrate*SYS_args['lrate_coe_sys'],minibatch_gpu_in:sched.minibatch_gpu,
                                    real_B_sys:real_B_sys_img}
                    )
                    _ = tf.get_default_session().run(
                        [D_opt_sys], 
                        feed_dict={real_B_sys: real_B_sys_img, lrate_in_sys: 0.0002,
                                    fake_B_sys: fake_B_sys_img[0], minibatch_gpu_in:sched.minibatch_gpu}
                    )
                    #NOTE: the sequence of updating networks follows cyclegan and stylegan
                tflib.run([D_train_op, Gs_update_op], feed_dict)
                if run_D_reg:
                    tflib.run(D_reg_op, feed_dict)

            # Slow path with gradient accumulation.
            else:
                for _round in rounds:
                    _g_loss, _ = tflib.run([G_loss, G_train_op], feed_dict)
                if run_G_reg:
                    for _round in rounds:
                        tflib.run(G_reg_op, feed_dict)
                # update G_a & D_sys via system loss
                if hier_training:
                    for _round in rounds:
                        # load minibatch
                        real_B_sys_img = training_set_sys.next()
                        while real_B_sys_img.shape[0] != sched.minibatch_gpu:
                            real_B_sys_img = training_set_sys.next()
                        # update G_a
                        _ = tf.get_default_session().run(
                            [G_opt_from_sys],
                            feed_dict={lrate_in_sys: sched.G_lrate*SYS_args['lrate_coe_sys'],minibatch_gpu_in:sched.minibatch_gpu,
                                        real_B_sys:real_B_sys_img}
                        )

                        #NOTE: test code, extract the gradients
                        # [_gra_agent] = tflib.run([_gra_a], feed_dict)
                        # [_gra_system] = tf.get_default_session().run(
                        #     [_gra_s],
                        #     feed_dict={minibatch_gpu_in:sched.minibatch_gpu,
                        #                 real_B_sys:real_B_sys_img}
                        # )
                    for _round in rounds:
                        # load minibatch
                        real_B_sys_img = training_set_sys.next()
                        while real_B_sys_img.shape[0] != sched.minibatch_gpu:
                            real_B_sys_img = training_set_sys.next()
                        # update D_sys
                        fake_B_sys_img = tf.get_default_session().run(
                            [output_G_sys],
                            feed_dict={minibatch_gpu_in:sched.minibatch_gpu,real_B_sys:real_B_sys_img}
                        )
                        _ = tf.get_default_session().run(
                            [D_opt_sys], 
                            feed_dict={real_B_sys:real_B_sys_img, lrate_in_sys: 0.0002,
                                        fake_B_sys: fake_B_sys_img[0], minibatch_gpu_in:sched.minibatch_gpu}
                        )
                tflib.run(Gs_update_op, feed_dict)
                for _round in rounds:
                    tflib.run(data_fetch_op, feed_dict)
                    tflib.run(D_train_op, feed_dict)
                if run_D_reg:
                    for _round in rounds:
                        tflib.run(D_reg_op, feed_dict)

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if cur_tick < 0 or cur_nimg >= tick_start_nimg + sched.tick_kimg * 1000 or done:
            cur_tick += 1
            tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
            tick_start_nimg = cur_nimg
            tick_time = dnnlib.RunContext.get().get_time_since_last_update()
            total_time = dnnlib.RunContext.get().get_time_since_start() + resume_time

            # Report progress.
            print('tick %-5d kimg %-8.1f lod %-5.2f minibatch %-4d time %-12s sec/tick %-7.1f sec/kimg %-7.2f maintenance %-6.1f gpumem %.1f' % (
                autosummary('Progress/tick', cur_tick),
                autosummary('Progress/kimg', cur_nimg / 1000.0),
                autosummary('Progress/lod', sched.lod),
                autosummary('Progress/minibatch', sched.minibatch_size),
                dnnlib.util.format_time(autosummary('Timing/total_sec', total_time)),
                autosummary('Timing/sec_per_tick', tick_time),
                autosummary('Timing/sec_per_kimg', tick_time / tick_kimg),
                autosummary('Timing/maintenance_sec', maintenance_time),
                autosummary('Resources/peak_gpu_mem_gb', peak_gpu_mem_op.eval() / 2**30)))
            autosummary('Timing/total_hours', total_time / (60.0 * 60.0))
            autosummary('Timing/total_days', total_time / (24.0 * 60.0 * 60.0))

            # Save snapshots.
            if image_snapshot_ticks is not None and (cur_tick % image_snapshot_ticks == 0 or done):
                print('g loss', _g_loss)
                grid_fakes = Gs.run(grid_latents, grid_labels, is_validation=True, minibatch_size=sched.minibatch_gpu)
                misc.save_image_grid(grid_fakes, dnnlib.make_run_dir_path('fakes%06d.png' % (cur_nimg // 1000)), drange=drange_net, grid_size=grid_size)

            if network_snapshot_ticks is not None and cur_tick % network_snapshot_ticks == 0 or done:
                pkl = dnnlib.make_run_dir_path('network-snapshot-%06d.pkl' % (cur_nimg // 1000))
                misc.save_pkl((G, D, Gs), pkl)
                metrics.run(pkl, run_dir=dnnlib.make_run_dir_path(), data_dir=dnnlib.convert_path(eval_data_dir), num_gpus=num_gpus, tf_config=tf_config, rho=rho)
            # Update summaries and RunContext.
            metrics.update_autosummaries()
            #tflib.autosummary.save_summaries(summary_log, cur_nimg)
            dnnlib.RunContext.get().update('%.2f' % sched.lod, cur_epoch=cur_nimg // 1000, max_epoch=total_kimg)
            maintenance_time = dnnlib.RunContext.get().get_last_update_interval() - tick_time

    # Save final snapshot.
    misc.save_pkl((G, D, Gs), dnnlib.make_run_dir_path('network-final.pkl'))

    # All done.
    summary_log.close()
    training_set.close()
    eval_set.close()
#----------------------------------------------------------------------------

