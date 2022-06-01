# Experiment for images
This folder details the Few-shot Image Generation experiment. To run the code, we provide the required environment (Docker). This model requires the checkpoints of other pre-trained models.

The implementation is built on [FSGAN](https://github.com/e-271/few-shot-gan) and StyleGAN2. We thank for their prior work. 

## Setup
### Requirements
- Hardware: GPU
- Deploy the Linux environment per the suggestion at official implmentation of StyleGAN2: https://github.com/NVlabs/stylegan2#requirements.
- Install dependencies in: `requirements.txt`
- Note the environment and code are tested on a local server (Intel(R) Xeon(R) Gold 5218R CPU, 2x Quadro RTX 6000, 128GB RAM)

### Data

This experiment requires:
- training set: minihorse_train_{5,10,30}
- test set: minihorse_test
- training set for the mixer: zebra

Please download minihorse_train and minihorse_test from the link: https://drive.google.com/drive/folders/10bW9s9BVvGCLiNdSD5nBb8DJdlHcpTPO?usp=sharing. 

We need to process them by following command: 

```
python dataset_tool.py \
create_from_images \
/path/to/target/tfds \
/path/to/source/folder \
--resolution 256

# example
python dataset_tool.py \
create_from_images \
./data/tfds/minihorse_train_5 \
./data/minihorse_train_5 \
--resolution 256    
```

For the dataset zebra, please download `horse2zebra.zip` here: https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/. Extract the file and move the foler `trainB` into path `./data/zebra`.

Prepare all three datasets before training the model.

### Checkpoints

To replicate experiments from the paper, you will need the checkpoint of the StyleGAN2 and the CycleGAN.

- Our networks start with pre-trained checkpoint pickle from vanilla StyleGAN2 `stylegan2-horse-config-f.pkl`, which can be downloaded from Drive here: [StyleGAN2 Checkpoints](https://drive.google.com/file/d/1irwWI291DolZhnQeW-ZyNWqZBjlWyJUn/view?usp=sharing). Save the checkpoint: `./checkpoint/stylegan2-horse-config-f.pkl`
- The code will download the pretrained CycleGAN automatically, please make sure the Internet access. If the code failed to download it, please download it manually: [CycleGAN checkpoints](https://drive.google.com/file/d/1RVIBe_h6ttvVTPslT-MvAz-4U76iY6kP/view?usp=sharing). Extract the zip file in folder: `./checkpoint/system_models`

## Training MGM

Then train MGM (our method) using this command:

Key argument: 
- config: `config-ada-sv-flat` to train a FSGAN, while `config-fd` to train a FreezeD
- hier: input this argument to train a MGM while not input to train a baseline

```
python run_training.py \
--config=config-ada-sv-flat \
--data-dir=/path/to/datasets \
--dataset-train=path/to/train \
--dataset-eval=path/to/eval \
--resume-pkl-dir=/path/to/pickles \
--total-kimg=30 \
--metrics=None \

# example
python run_training.py \
--config=config-ada-sv-flat \
--data-dir=./data/tfds \
--dataset-train=/minihorse_train_5 \
--dataset-eval=/minihorse_test \
--resume-pkl-dir=./checkpoint/ \
--resume-pkl=stylegan2-horse-config-f.pkl  \
--total-kimg=30 \
--metrics=fid1k \
--hier # this argument enables to train a MGM model, comment it to train the baseline
```

The example command can provide the result of FSGAN-A with 5-shot training data.

## Image generation

To generate additional samples from a trained model:

```
python run_generator.py generate-images --network=/path/to/network/pickle --seeds=0-100
```

## License

As a modification of the FSGAN and the official StyleGAN2 code, this work inherits the Nvidia Source Code License-NC. To view a copy of this license, visit https://nvlabs.github.io/stylegan2/license.html
