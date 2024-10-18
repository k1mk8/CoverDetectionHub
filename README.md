# CoverDetectionHub

## Project description

The aim of the project is to prepare a framework for music cover detection. The main assumption is to implement a "hub" that lets further researchers carry out various experiments in this field as well as compare different Music Information Retrieval (MIR) methods.

## Bibliography review

### Datasets and benchmarks

| **Dataset**                | **Details**                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------|
| Da-TACOS – <br>Dataset for Cover Song Identification<br> and Understanding <br> <li>[Da-TACOS Dataset Paper](https://archives.ismir.net/ismir2019/paper/000038.pdf)</li>  <br> <li>[Da-TACOS GitHub Repository](https://github.com/MTG/da-tacos)</li>  | Two subsets: <br> 1. Benchmark Subset (15,000 songs) <br> 2. Cover Analysis Subset (10,000 songs) <br> <br> <li>Annotations obtained with API from SecondHandSongs.com</li> <br> <li>Features extracted from MP3 audio files encoded at 44.1 kHz sample rate</li> <br> <li>**No audio files included, only pre-extracted features and metadata**</li> <br> <li>7 state-of-the-art CSI algorithms benchmarked on the Benchmark Subset</li> <br> <li>Cover Analysis Subset used to study modifiable musical characteristics  </li>   <br> Thoughts: This dataset has become a classic benchamark for testing CSI systems. Moreover, authors of the paper and dataset also provide a  framework for feature extraction and benchmarking - [acoss: Audio Cover Song Suite](https://github.com/furkanyesiler/acoss). 'acoss' includes a standard feature extraction framework with audio features for CSI task and open source implementations of seven CSI algorithms. It was designed to facilitate the future work in this line of research. Although dataset in relatively new (2019), both repositories have not been updated since 5 years ago and considering how rapidly MIR domain develops - 5 years is a lot. That is why our project can be an attempt to create a refreshed and modern version of this framework. It would include state-of-the-art methods with hopefully additional datasets to test them.


## Environment setup
Our proposed technology stack is based on Python, considering its great capabilities for working with data in an easy way.

TO ADD

## Config file structure

TODO

## Proposed experiments

TODO

## Dataset for experiments

We intend to use Da-TACOS dataset (https://github.com/MTG/da-tacos) because of its versatility, decent size and excellent metadata structure. It is organised into cliques that gather an original performance along with its covers, which fits perfectly into our needs.

In future extensions, it is possible to utilize a dataset delivered by the Polish Society of Authors ZAiKS.

## Project schedule
### W1 (14-20.10.2024)

Gathering literature, preparing design proposal, tools selection, selection of dataset

### W2 (21-27.10.2024)

Preparing the environment, choice for the models, initial dataset preprocessing

### W3-W4 (28.10-10.11.2024)

Implementation of the first functional prototype, including training at least one model and minimal GUI

### W5-W6 (11-24.11.2024)

First results evaluation, implementing improvements, training and adding subsequent models

### W7 (25.11-1.12.2024)

Automated tests design, training of remaining models

### W8-W9 (02-15.12.2024)

Evaluation of the results, improving GUI, re-training the models if necessary

### W10-W12 (16.12.2024-05.01.2025)

Working on the final presentation, tests, gathering final results, Xmas chill (optional)

### W13 (06-13.01.2025)

Final results evaluation, preparation of the paper

### W14 (13-19.01.2025)

Public project presentation
