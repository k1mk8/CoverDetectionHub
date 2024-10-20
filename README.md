# CoverDetectionHub

## Project description

The aim of the project is to prepare a framework for music cover detection. The main assumption is to implement a "hub" that lets further researchers carry out various experiments in this field as well as compare different Music Information Retrieval (MIR) methods.

## Bibliography review

### Cover song detection methods

| **Paper**                | **Notes**                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------|
| [CoverHunter: Cover Song Identification with Refined Attention and Alignments](https://arxiv.org/pdf/2306.09025) | Paper proposes new method for CSI task called CoverHunter. It explores features deeper with refined attention and alignments. It has 3 crucial modules: 1) Convolution-augumented transformer; 2) Attention-based time pooling module; 3) A novel training scheme. Authors share their excellent results, beating all existing methods at this time. They test in on benchmarks like DaTacos and SHS100K. In general, they propose a state-of-the-art model, which is definitely one of the current best, which is why it is worth including it in our project. PyTorch implementation of this method can be found in repository [CoverHunter](https://github.com/Liu-Feng-deeplearning/CoverHunter), along with checkpoints of pretrained models. |
| [BYTECOVER: COVER SONG IDENTIFICATION VIA MULTI-LOSS TRAINING](https://arxiv.org/pdf/2010.14022v2) | This paper from 2021 introduces new feature learning method for song cover identification task. It is build on classical ResNet model with improvements designed for CSI. In a set of experiments, authors demonstrate its effectiveness and efficiency. They evaluate the method on multiple datasets including DaTacos. In the repository [bytecover](https://github.com/Orfium/bytecover), there is a shared implementation of this method with the best-trained model checkpoints. Thoughts: There is no transformer in a method, which may imply worse results than CoverHunter. |
|[ESSENTIA: AN AUDIO ANALYSIS LIBRARY FOR MUSIC INFORMATION RETRIEVAL](https://www.researchgate.netpublication/256104772_ESSENTIA_an_Audio_Analysis_Library_for_Music_Information_Retrieval) | This paper describes a framework for multiple MIR applications. This tool consists of a number of reconfigurable modules that come in handy for researchers. For our case, an interesting approach is to use the harmonic pitch class profile and the chroma features of audio signals to calculate the similarity of two tracks. This model is very basic and well-known; therefore, it will serve as a reference.

### Datasets and benchmarks

| **Dataset**                | **Details**                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------|
| Da-TACOS – <br>Dataset for Cover Song Identification<br> and Understanding <br> <li>[Da-TACOS Dataset Paper](https://archives.ismir.net/ismir2019/paper/000038.pdf)</li>  <br> <li>[Da-TACOS GitHub Repository](https://github.com/MTG/da-tacos)</li>  | Two subsets: <br> 1. Benchmark Subset (15,000 songs) <br> 2. Cover Analysis Subset (10,000 songs) <br> <br> <li>Annotations obtained with API from SecondHandSongs.com</li> <br> <li>Features extracted from MP3 audio files encoded at 44.1 kHz sample rate</li> <br> <li>**No audio files included, only pre-extracted features and metadata**</li> <br> <li>7 state-of-the-art CSI algorithms benchmarked on the Benchmark Subset</li> <br> <li>Cover Analysis Subset used to study modifiable musical characteristics  </li>   <br> Thoughts: This dataset has become a classic benchamark for testing CSI systems. Moreover, authors of the paper, along with the  dataset, also provided a  framework for feature extraction and benchmarking - [acoss: Audio Cover Song Suite](https://github.com/furkanyesiler/acoss). 'acoss' includes a standard feature extraction framework with audio features for CSI task and open source implementations of seven CSI algorithms. It was designed to facilitate the future work in this line of research. Although dataset in relatively new (2019), both repositories have not been updated since 5 years ago and considering how rapidly MIR domain develops - 5 years is a lot. That is why our project can be an attempt to create a refreshed and modern version of this framework. It would include state-of-the-art methods with hopefully additional datasets to test them. |
|  <br> [Covers80](http://labrosa.ee.columbia.edu/projects/coversongs/covers80/) |   <li>The dataset contains 80 songs, with 2 different performances of each song by different artists (160 tracks in total).  </li>   <br> <li>All audio files are encoded as 32 kbps MP3 (mono, 16 kHz sampling rate, bandwidth limited to 7 kHz). </li>  <br> Thoughts: We will not use the Covers80 dataset as primary dataset  because it is relatively small and is old (2007). Additionally, the audio files are of low quality (32 kbps, 16 kHz mono).The dataset was assembled somewhat randomly, and it may not provide the diversity or representativeness. However, it has become a CSI systems benchmark, that is why, if we have enough time, we will try to include it in out project. <br> Dataset appeared in a paper [THE 2007 LABROSA COVER SONG DETECTION SYSTEM](http://labrosa.ee.columbia.edu/~dpwe/pubs/EllisC07-covers.pdf). |
| [SHS100K](http://millionsongdataset.com/secondhand/) | <li> Contains metadata and audio features for a large number of songs and their covers. </li> <li> Includes a diverse range of musical genres </li>   <li> Metadata: song title, artist, release year </li> <li> Audio features: chroma, key, tempo, and others related to music structure and timbre. </li> <br> Thoughts: Another benchmark for cover detection task. We will consider it as a secodary dataset. |


## Technology stack
Main technologies in use:
- **Python**: Our proposed technology stack is based on Python, considering its great capabilities for working with data in an easy way. 
- **Gradio**: User interface will be implemented in Gradio library, because it is a very convenient tool for a fast prototype building.
- **Numpy**: Library for maths operations.
- **PyTorch**: Deep learning library.
- **essentia**: A versatile tool for MIR operations
- **venv** (or other tool): For making the project portable in an easy way

Probably there will appear more libraries, strictly from MIR domain, which we will get to know during project development.


## Planned functionality of the hub
- Testing CSI methods on Da-TACOS dataset
- Choosing between methods
- Report on evaluation and calculated metrics

## Dataset for experiments

We intend to use Da-TACOS dataset (https://github.com/MTG/da-tacos) because of its versatility, decent size and excellent metadata structure. It is organised into cliques that gather an original performance along with its covers, which fits perfectly into our needs.

In future extensions, utilising a dataset delivered by the Polish Society of Authors ZAiKS is possible.

## Experiment scope

In the initial phase of the project, it is intended to perform 1-1 comparisons for checking the similarity rate.

For the evaluation of analysed methods, there are proposed 2 experiments:

- synthetic generation of samples, where a defined percentage of an original piece is injected into a totally different sample (Batlle-Roca et al., https://arxiv.org/pdf/2407.14364)
- direct comparison between two samples, where 2 full samples are processed. 

## Computing resources
To train and run deep learning models, our project will need GPU devices. So far, we gathered two seperate units: RTX3090Ti and RTX2060.

## Project schedule
### W1 (14-20.10.2024)

- [x] Gathering literature
- [x] preparing design proposal
- [x] tools selection
- [x] selection of dataset

### W2 (21-27.10.2024)

- [ ] Preparing the environment
- [ ] choice for the models
- [ ]  initial dataset preprocessing

### W3-W4 (28.10-10.11.2024)

- [ ] Implementation of the first functional prototype
- [ ] including training at least one model
- [ ] minimal GUI

### W5-W6 (11-24.11.2024)

- [ ] First results evaluation
- [ ] implementing improvements
- [ ] training
- [ ] adding subsequent models

### W7 (25.11-1.12.2024)

- [ ] Automated tests design
- [ ] training of remaining models

### W8-W9 (02-15.12.2024)

- [ ] Evaluation of the results
- [ ] improving GUI
- [ ] re-training the models if necessary

### W10-W12 (16.12.2024-05.01.2025)

- [ ] Working on the final presentation
- [ ] tests
- [ ] gathering final results, Bożenarodzeniowy chill (optional)

### W13 (06-13.01.2025)

- [ ] Final results evaluation
- [ ] preparation of the paper (?)

### W14 (13-19.01.2025)

:tada: Public project presentation :tada:
