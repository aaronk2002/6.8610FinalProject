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



## Trainig

To train the model and generate some music, run the following, which will train the transformer model and save it to the trained_models folder, and it will then generate music from that model. To change the hyperparameters used to train the model, change the values in config/addison.yml and config/save.yml (Addison is a pseudonym of Aaron).

```bash
$ sh bash_scripts/run.sh
```
