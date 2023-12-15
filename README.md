# Music Transformer Probing

This repository builds on top of the following [repository](https://github.com/jason9693/MusicTransformer-pytorch.git) by [jason9693](https://github.com/jason9693)

## Simple Start ( Repository Setting )

```bash
$ git clone https://github.com/aaronk2002/6.8610FinalProject
$ cd 6.8610FinalProject
$ git clone https://github.com/jason9693/midi-neural-processor.git
$ mv midi-neural-processor midi_processor
```



## Dataset Acquisition

Download the dataset from this [link](https://www.kaggle.com/datasets/kritanjalijain/maestropianomidi/) and put the downloaded folder in the dataset folder

```bash
$ sh bash_scripts/preprocess.sh
```



## Training and Evaluating Transformer Performance

To train the model and generate some music, run the following, which will train the transformer model and save it to the trained_models folder, and it will then generate music from that model. To change the hyperparameters used to train the model, change the values in config/addison.yml and config/save.yml (Addison is a pseudonym of Aaron).

```bash
$ sh bash_scripts/run.sh
```

To evaluate, rename the model to {number of decoder layers}.pth and run

```bash
$ python evaluation.py --layers {number of decoder layers} --N {number of samples per batches} --M {number of batches}
```



## Probe Dataset Generation

To generate dataset for training the probe models, run the following in the probing folder

```bash
$ python generate_probe_data.py --model {transformer model path} --save {dataset save path}
```

A recommended way to do this is to rename your model paths to {number of decoder layers}.pth in the trained_model folder, training the models with 4, 6, and 8 layers, and run

```bash
$ sh bash_scripts/probe_generate.sh
```



## Probe Training and Evaluation

To train and evaluate the probes, choose first the transformer model that you want to probe, and create the dataset using the previous step. Name the dataset {number of decoder layers in model}-layers-probe.pth and put it in the dataset folder. Then, run the following to train and evaluate probes

```bash
$ python probing/train_probes.py --layers {number of models in the transformer} --task {type of task: control, key, or composer} --lr {learning rate} --epochs {number of epochs}
$ python probing/evaluate_probes.py --layers {number of models in the transformer} --task {type of task: control, key, or composer} --lr {learning rate} --epochs {number of epochs}
```
