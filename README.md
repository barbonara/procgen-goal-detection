# goal-extraction-procgen

## Installation

Note: We had trouble getting procgen to work on our local machines (Mac with M1 chip). We decided to use a Virtual Machine running Debian, Debian GNU/Linux, 11 (bullseye), amd64.

1. Go to procgen-custom-maze directory
2. Create a conda environment from the .yml file: `conda env create -f environment.yml -n <custom env name>`
3. Switch to the newly created environment
4. Run `pip install -e .` inside the procgen-custom-maze directory
5. Go to procgen-tools-custom-mazes directory
6. Run `pip install -e .` inside the procgen-tools-custom-mazes directory

I had to install `sudo apt-get install libgl1-mesa-dev` at some point


Models can be found here: https://drive.google.com/drive/folders/1Ig7bzRlieyYFcdKL_PM-guSWR8WryDOL 

TO DO: create a drive with our own models
