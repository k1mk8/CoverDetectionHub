# CoverDetectionHub 2025L continuation - Design Proposal

------------------------

## Autors

***Karol Kasperek***

***Wojciech Bożek***

***Mateusz Węcławski***



## Project description
Aim of the actual term is to expand functionalities of  **CoverDetectionHub** created on previous term by Kamil Szczepanik, Dawid Ruciński, Piotr Kitłowski.
 
Actual state of project contains framework and “hub” for **music cover identification** enabling researchers to compare various CSI methods and run experiments through a user-friendly Gradio interface. 

Purposes for term 25L is to try to develop future challenges from previous implementation and add new functionalities which we will describe later.



## Planned functionalities

- obtaining Da-Tacos dataset and possibly perform training and evaluation on it
- improving the [LyriCover](https://github.com/DawidRucinski/LyriCover) model for more sophisticated audio features exctraction; training on a larger subset of SHS100k after improving performance of the model, speed up calculating of actual model
- performing more experiments, similar to "Injected Abracadabra" or others found in the literature
- augmentations
- model for generating covers from given track (if we have capacity to do that - not included in schedule)


## Technology stack
Main technologies in use:
- **Python**: Our proposed technology stack is based on Python, considering its great capabilities for working with data in an easy way.
- **PyTorch**: Deep learning library.
- **Gradio**: User interface will be implemented in Gradio library, because it is a very convenient tool for a fast prototype building.
- **Numpy**: Library for maths operations.
- **Librosa**: Used for audio loading and some feature extraction (MFCC, spectral centroid) in the simpler comparison methods.
- **venv** (or other tool): For making the project portable in an easy way
- **OpenAI Whisper**: Used by Lyricover to transcribe lyrics and measure similarity in lyric space.

## Project proposed schedule

- **Week 1 (20.02 - 23.02)**:	Create team and discord channel
- **Week 2 (26.02 - 01.03)**:	Topic selection and overview of projects from previous terms
- **Week 3 (04.03 - 08.03)**:	Create entrance design proposal
- **Week 4 (11.03 - 15.03)**:	Design proposal refinement and literature study
- **Week 5-6 (18.03 - 29.03)**:	Understand of Da-Tacos dataset and use it for testing
- **Week 7 (01.04 - 05.04)**:	Dataset augmentations for better coverage of diffrent scenarios
- **Week 8-9 (08.04 - 19.04)**:	Improving the LyriCover model and testing
- **Week 10-11 (22.04 -3.05)**:	Further improvig of LyriCover and training on larger dataset
- **Week 12 (06.05 - 10.05)**:	Update and improve UI for new and previous model, fix for created model
- **Week 13 (13.05 - 17.05)**:	Prepare tests for existing code
- **Week 14 (20.05 - 24.05)**:	Evaluation of the results and prepare for final presentation
- **Week 15 (27.05 - 31.05)**:   Final presentation and create recording

*It's only demonstration of schedule. Work during weeks may change in case of troubles or low capacity due to other projects.


-----------------------

## Bibliography

#TODO 