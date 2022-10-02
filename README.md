# aintelope

We operationalize conjectures from Steven Byrnes’ 
[“Reverse-engineer human social instincts”](https://www.lesswrong.com/s/HzcM2dkCq7fwXBej8/p/tj8AC3vhTnBywdZoA)
research program and extend existing research into brain-like AI. 
We simulate agents with reinforcement learning over selected cue-response patterns 
in environments that could give rise to humanl-iike complex behaviors. 
To do so we select cue-responses from a pre-existing list of more than 
60 candidates for human affects and instincts from affective neuroscience 
and other sources including original patterns. 
Cue-responses are conjectured to form hierarchies and the project will start 
the simulation of lower-level patterns first. 
We intend to verify the general principles and the ability of our software to 
model and simulate the agents and the environment to a sufficient degree.


Working document:
https://docs.google.com/document/d/1qc6a3MY2_guCZH8XJjutpaASNE7Zy6O5z1gblrfPemk/edit#

## Installation

### Poetry

Dependencies are managed via `poetry`. See installation instructions
[here](https://python-poetry.org/docs/#installation).

To install the python dependencies run `poetry install`.
To activate the environment via poetry run `poetry shell`. Alternatively, most
commands can also be executed without an active python environment via
`poetry run <command>`.

To execute the current trainer run `poetry run aintelope`.

### Code formatting

To autoformat python files you can use [`black`](https://github.com/psf/black).
To automatically sort the imports you can run
[`isort .`](https://github.com/PyCQA/isort) from the root level of the project.

See the `Makefile` for further instructions.

## Running

Try `/bin/python3 aintelope/environments/savanna.py`.


# Cygwin

We have not managed to get aintelope to build under Windows/Cygwin!
You need at least the following prerequisites:

* Python: Cygwin modules Python 3.7 including python3-devel (but no 3.9+ available)
* Pytorch: https://github.com/KoichiYasuoka/CygTorch

Problems:
* Windows Python versions don't work with the poetry commands.
* poetry shell doesn't work with paths with spaces.
* Even with CygTorch pytorch-lightning can't find a Torch version.
* other
