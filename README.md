# CMR-IA: A Context-Maintenance and Retrieval Model for Item and Associative Memory

CMR-IA is a computational model of memory, belonging to the family of Context-Maintenance and Retrieval (CMR) Models. CMR-IA aims to provide a mechanistic explanation for memory of items and associations, which concerns tasks including item recognition, associative recognition, and cued recall. 

This is a project in the Computational Memory Lab at Upenn. It is maintained by Beige Jin. Many of the codes are inherited from [Pazdera & Kahana (2022)](https://github.com/jpazdera/PazdKaha22), especially the big structure of the model and the particle swarm optimization. But please be cautious that major modifications are also in multiple aspects, including how to use the model. We've marked the changes in the code with comments like `[bj]` and `[Newly added]`.

The following is an overview of the two directories included herein (for more details, please see the README files located within each directory):
- `Analysis/`: Jupyter notebooks for running simulations, analyzing data, and generating figures.
- `Modeling/`: Python/Cython code for the CMR-IA model and particle swarm optimization.

## Quick Start

(1A) If you do not already have a conda environment set up, use the provided shell script in `Modeling/CMR_IA/` to set up an environment, build and install CMR-IA and its required packages, and set up a Jupyter kernel for the environment:

~~~
cd Modeling/CMR_IA
bash setup_env.sh
~~~

(1B) If you already have an environment in Anaconda, simply activate it (replace ENV_NAME with the name of your environment), make sure CMR-IA's dependencies are installed, and then run its installation script in `Modeling/CMR_IA/`:

~~~
source activate ENV_NAME
cd Modeling/CMR_IA
pip install -r requirements.txt
python setup.py install
~~~

Note: please ignore the error: Could not find suitable distribution for Requirement.parse('CMR-IA==0.0.0').

(2) Regardless of which of the two methods you used to install CMR, you should now be able to import it into Python from anywhere using the following line:

~~~
import CMR_IA as cmr
~~~

(3) Once you have imported it, you can use the functions from the package just like you would use any other package in Python. You can run the simulation notebooks in `Analysis/` to reproduce the figures in the paper.

(4) For developers, any time you change or update CMR_IA, you will need to rerun the following line in order to rebuild the code and update your installation:

~~~
python setup_cmr.py install 
~~~

(5) Note that the optimization algorithms will not be installed as part of the package. If you wish to use these scripts, you will need to modify them as needed to work with your research project, and then run the files directly (see README in `Modeling/`).

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
