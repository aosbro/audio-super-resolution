# audio-super-resolution
This repository contains the code that was implemented in the context of an optional semester project at the 
[VITA](https://www.epfl.ch/labs/vita/) lab credited 8 ECTS. 

The objective of the project is to adapt the frameworks of [Generative Adversarial Networks](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
to the field of audio which in contrast to images has received relatively few attention in the recent years. The task
that we aim to solve lies at the frontier of super-resolution and style transfer. Formally, we desire to recover an
original signal from an observed altered version of it. The alterations could be the effect of the recording with a
cheap microphone and in a non-anechoic environment on the original audio content. A more style transfer-ish type of
application would be a change of instrument in the mapping from the original to the altered version of the signal. The
second type of transformation presents the strong advantage that it is relatively uniformly present throughout the whole
signal whereas the level of distortion of the first transformation would typically depend of the local content of the 
signal (amplitude/frequency). Considering the scarcity of the literature on that topic, we turned our attention to the 
second type of transformation.
 
## Setup
All the code present in this repository was implemented and tested on Ubuntu 18.04 and Python 3.6.
To create a conda environment with the minimal requirements proceed as follows from a terminal:
```
# Make the script executable
chmod +x create_conda_env.sh

# Launch the script in interactive mode (it will ask for the name of the environment)
bash -i create_conda_env.sh

# Activate the environment
conda activate <env_name>
```
Once the environment is not needed anymore it can be removed as follows:
```
# Remove the environment
conda remove --name <env_name> --all
```
In addition to the Python dependencies, it is also needed to install [TiMidity++](http://ccrma.stanford.edu/planetccrma/man/man1/timidity.1.html)
which is the software used to process the .midi files. On Ubuntu it can be installed from a terminal as follows:
```
# Install TiMidity++
sudo apt-get install -y timidity

# Example command to convert a .midi file to .wav
timidity input.midi -Ow -o output.wav
```

## Dataset 
The dataset used in this project relies on the [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro) which
contains over 200 hours of piano solos sampled at frequency in the range 44.1-48 kHz. In addition to the audio files it 
further contains a .midi file for each .wav file. The [MIDI](https://www.noterepeat.com/articles/how-to/213-midi-basics-common-terms-explained)
format is an event based communication standard that allows different musical peripherals to communicate together. For 
instance if an electric piano is being played, each time a key is "pressed" a message which contains the note and the 
velocity at which it was pressed is sent. Later, when the key is "released" another message with the same note, but 
with zero velocity, is sent to stop the note from being played. In addition to the note message there also exists 
control messages which, as the name suggests, allows the player to "control" how the notes should be played. In the 
example of the piano, the controls are limited to the pedals actions. Finally, another interesting type of messages is 
the program message which store the information related to the instrument to play.

#### Create the dataset files
To create the dataset from the .midi files, we use the [Mido](https://mido.readthedocs.io/en/latest/) library which 
allows us to parse the messages of a given .midi file and to modify them as needed. In the context of this project the 
transformation applied to original signal to get the observed signal consists of a change of instrument from 
"Acoustic Grand Piano" to "Rhodes Piano". The symbolic code for each instrument can be found [here](http://www.ccarh.org/courses/253/handout/gminstruments/). 
Throughout the code, the convention is adopted to refer to original signal as ``target`` and the observed signal as 
``input``. The overall objective of the models is to learn the mapping from the ``input`` to the ``target``, it is 
therefore needed to build a dataset of tuples of ``(input, target)`` signals containing a fixed number of samples, 
``window_length``. More can be found in the report on how to choose this parameter, the default value is currently set
 to 8192 samples per window.
 
The script ``create_maestro_file.py`` can be used to create the files needed for the train, test and validation phases 
of the experiments. It can either create one .npy file for each phase in which case each file will have shape 
``[N, 2, window_length]``. On the other hand if the .hdf5 format is chosen, a single file will be created. The sub-files
 are first accessed by phase (``'train'``/``'test'``/``'valid'``) and then by type (``'input'``/``'target'``):
```
# Example access of data from .hdf5 file
with h5py.File(hdf5_filepath, 'r') as hdf:
        x_input = hdf['train']['input'][index]
``` 

Note that choosing the .hdf5 file format is advised as it will work with an arbitrary large number of signals as the
data is retrieved from the disk and does not need to fit entirely in ram. To mitigate speed problem a "pseudo cache" in 
ram is implemented. More can be found in the docstring of the class ``DatasetMaestroHDF`` in file ``datasets.py``. 
Every argument to pass when calling the ``create_maestro_file.py`` script have a default value and have an explanation 
in the argument parser.
```
# Display the help menu
python3 create_maestro_file.py --help
```

## Training the models
Each model can be trained through its own file. The training is managed by a trainer class which stores all the 
attributes and parameters needed to be able to train a model over multiple sessions. Every argument to pass when 
calling the ``train_<model_name>.py`` script have a default value and have an explanation in the argument parser.
```
# Display the help menu
python3 train_<model_name>.py --help
```

## Generating a track with a pre-trained model
Once a model is trained, it can be used to generate a part of track in order to assess its performance subjectively.
To do so, the file ``generate_single_track.py`` can be used. Every argument to pass when calling the 
``generate_single_track.py`` script have a default value and have an explanation in the argument parser.
```
# Display the help menu
python3 generate_single_track.py --help 
```