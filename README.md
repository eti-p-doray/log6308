# LOG6308 Project
Embeddings for Netflix prize Challenge

We implemented a few techniques in TensorFlow in order to try and compare their performance:

- Latent factors in `netflix_0.0_latent.py`
- Bezier curve on temporal component added to latent factorization in `netflix_0.1_latent.py`
- Neural network (not successful) in `netflix_1.0_nn.py`
- Gaussean kernel trick in `netflix_2.0_kernel.py`
- Word vectors on titles in `netflix_3.0_word2vec.py`
- Word vectors and latent factorization hybrid in `netflix_3.1_word2vec`

## Documentation

See presentation-log6308.pdf for presentation slides (french) and rapport-log6308.pdf for accompanying comments.

## Installation

As we didn't make a build configuration, you will need to manually install the requirements and download the data in order to be able to run the project.

### Python3

Download it from [here](https://www.python.org/downloads/). The following installations are python packages/libraries that can be installed with `pip3`.

#### TensorFlow

You can install a version using the CPU with `pip3 install tensorflow`. For more detailed installation instructions, visit [their website](https://www.tensorflow.org/install/)

#### Numpy

We use numpy arrays everywhere to optimize size. Install it with `pip3 install numpy`.

#### Gensim

This library is used for the scrips using word2vec (`netflix_3.0_word2vec.py` and `netflix_3.1_word2vec.py`). Install it with `pip3 install gensim`.

### Data

We haven't included any data in this project, you'll need to get your hand on it yourself.

#### Netflix Prize data

Note that we have provided a small subset of the data in binary form. If used, this step can be skipped.
You will first need to find the dataset given with the Netflix Prize. At the time of this project's creation, you could find it at the top of a Google search. Unzip it's content at the root of this project, such as you would have a `nf_prize_dataset` folder at the root, which would contain a `training_set` folder. Once that is done, execute `python3 netflix_data.py` in order to make numpy array files with this data (and they're the ones to be used with the rest of the scripts) :

- `nf_prize_dataset/nf_prize.npz` : Array with all the data contained in the training set. If you rerun `netflix_data.py` while it exists, it'll be imported and used instead of being regenerated.
- `nf_prize_dataset/nf_probe.npz` : Array containing the probe dataset as indicated by `nf_prize_dataset/probe.txt`, extracted from the training set. We use it as our main validation set, since the official validation set (`nf_prize_dataset/qualifying.txt`) was made to be tested on Netflix's server and therefore its ratings isn't contained in the Netflix Prize's dataset.
- `nf_prize_dataset/nf_training.npz` : Array containing the training set without the data of the probe set (as indicated by `nf_prize_dataset/probe.txt`). Should be used as the training set when using `nf_prize_dataset/nf_probe.npz` as validation set.
- `nf_prize_dataset/small_nf_prize.npz` : Array containing a dense subset of 5000 users and 250 movies extracted from the dataset.
- `nf_prize_dataset/small_nf_probel.npz` : Array containing the members of the probe subset contained in the dense subset from `nf_prize_dataset/small_nf_prize.npz`. Can be used as a validation set on a smaller dataset together with `nf_prize_dataset/small_nf_training.npz`.
- `nf_prize_dataset/small_nf_training.npz` : Array containing the small training set from `nf_prize_dataset/small_nf_prize.npz` without the data from the small probe set (`nf_prize_dataset/small_nf_probel.npz`). Should be used as a training set together with `nf_prize_dataset/small_nf_probel.npz` as the validation set.

Finally, make a `log` folder at the root of the project : the output of the scripts will be put there. If you plan to use TensorBoard, you may want to copy `nf_prize_dataset/movie_titles.tsv` in this folder.

#### Google word2vec

Two of our scripts use word vectors to make predictions. For that, we have used a pretrained Google word2vec model, loaded with gensim. You can find the link for such pretrained model on [https://code.google.com/archive/p/word2vec/](https://code.google.com/archive/p/word2vec/) (find the section called "Pre-trained word and phrase vectors"). Once downloaded, unzip it and put the resulting `GoogleNews-vectors-negative300.bin` file in a `word2vec` folder at the root of this project.

## Training

Each script can be run without arguments to use default model configuration. Scripts are provided with help for how to use optional command line arguments. Here's an exemple of how to run model netflix_0.0_latent:

`python3 netflix_0.0_latent.py`

This will run model training and validation. Progress is written in console and optionally can be saved to log/model_name/ as a .csv with `--log_perf` flag. Model state is always saved to log/model_name/ folder and can be visualized with tensorboad.

We have provided a small subset of the data in binary form. To use it when running models, file inputs paths need to be provided:

`python3 netflix_0.0_latent.py -i nf_prize_dataset/small_nf_training.npz -t nf_prize_dataset/small_nf_probe.npz`
