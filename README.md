# audio-super-resolution

## Abstract 

## Method Overview

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
Once you do not need the environment anymore it can be removed as follows:
```
# Remove the environment
conda remove --name <env_name> --all
```
In addition to the Python dependencies, it is also needed to install [TiMidity++](http://ccrma.stanford.edu/planetccrma/man/man1/timidity.1.html)
which is the software used to process the .midi files. On Ubuntu it can be installed from a terminal as follows:
```
# Install TiMidity++
sudo apt-get install -y timidity
```
## Future work
