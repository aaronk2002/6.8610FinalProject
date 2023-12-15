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

```bash
$ python train.py -c {config yml file 1} {config yml file 2} ... -m {model_dir}
```



## Hyper Parameter

* learning rate : 0.0001
* head size : 4
* number of layers : 6
* seqence length : 2048
* embedding dim : 256 (dh = 256 / 4 = 64)
* batch size : 2



## Result

-  Baseline Transformer ( Green, Gray Color ) vs Music Transformer ( Blue, Red Color )

* Loss

  ![loss](readme_src/loss.svg)

* Accuracy

  ![accuracy](readme_src/accuracy.svg)



## Generate Music

```bash
$ python generate.py -c {config yml file 1} {config yml file 2} -m {model_dir}
```




## Generated Samples ( Youtube Link )

* click the image.

  [<img src="readme_src/sample_meta.jpeg" width="400"/>](https://www.youtube.com/watch?v=n6pi7QJ6nvk&list=PLVopZAnUrGWrbIkLGB3bz5nitWThIueS2)
