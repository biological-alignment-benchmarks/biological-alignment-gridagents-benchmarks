# AST experiments (title pending)

Description pending.

## Project setup

This readme contains instructions for both Linux and Windows installation. Windows installation instructions are located after Linux installation instructions.

### Installation under Linux

The project installation is managed via `make` and `pip`. Please see the respective commands in the `Makefile`. To setup the environment follow these steps:

1. Install CPython. The code is tested with Python version 3.10.10. We do not recommend using Conda package manager. 

Under Linux, run the following commands:

`sudo add-apt-repository ppa:deadsnakes/ppa`
<br>`sudo apt update`
<br>`sudo apt install python3.10 python3.10-dev python3.10-venv`
<br>`sudo apt install curl`
<br>`sudo curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10`

2. Get the code from repo:

`sudo apt install git-all`
<br>Run `git clone https://github.com/biological-alignment-benchmarks/biological-alignment-gridworlds-benchmarks.git`
<br>Run `cd biological-compatibility-benchmarks`

3. Create a virtual python environment:

`make venv-310`
<br>`source venv_aintelope/bin/activate`

4. Install dependencies:

`sudo apt update`
<br>`sudo apt install build-essential`
<br>`make install`

5. If you use VSCode, then set up your launch configurations file:

`cp .vscode/launch.json.template .vscode/launch.json`

Edit the launch.json so that the PYTHONPATH variable points to the folder where you downloaded the repo and installed virtual environment:

replace all
<br>//"PYTHONPATH": "your_path_here"
<br>with
<br>"PYTHONPATH": "your_local_repo_path"

6. For development and testing:

* Install development dependencies: `make install-dev`
* Run tests: `make tests-local`

7. Location of an example agent you can use as a template for building your custom agent: 
[`aintelope/agents/example_agent.py`](aintelope/agents/example_agent.py)


### Installation under Windows

1. Install CPython from python.org. The code is tested with Python version 3.10.10. We do not recommend using Conda package manager.

You can download the latest installer from https://www.python.org/downloads/release/python-31010/ or if you want to download a newer 3.10.x version then from https://github.com/adang1345/PythonWindows

2. Get the code from repo:
* Install Git from https://gitforwindows.org/
* Open command prompt and navigate top the folder you want to use for repo
* Run `git clone https://github.com/biological-alignment-benchmarks/biological-alignment-gridworlds-benchmarks.git`
* Run `cd biological-compatibility-benchmarks`

3. Create a virtual python environment by running: 
<br>3.1. To activate VirtualEnv with Python 3.10:
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`virtualenv -p python3.10 venv_aintelope` 
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(or if you want to use your default Python version: 
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`python -m venv venv_aintelope`)
<br>3.2. `venv_aintelope\scripts\activate`

4. Install dependencies by running:
<br>`pip uninstall -y ai_safety_gridworlds >nul 2>&1`
<br>`pip install -r requirements/api.txt`

5. If you use VSCode, then set up your launch configurations file:

`copy .vscode\launch.json.template .vscode\launch.json`

Edit the launch.json so that the PYTHONPATH variable points to the folder where you downloaded the repo and installed virtual environment:

replace all
<br>//"PYTHONPATH": "your_path_here"
<br>with
<br>"PYTHONPATH": "your_local_repo_path"

6. For development and testing:

* Install development dependencies: `pip install -r requirements/dev.txt`
* Run tests: `python -m pytest --tb=native --cov="aintelope tests"`

7. Location of an example agent you can use as a template for building your custom agent: 
[`aintelope\agents\example_agent.py`](aintelope/agents/example_agent.py)


### Setting up the LLM API access

Set environment variable:
`OPENAI_API_KEY`.

Ensure you have loaded credits on your OpenAI API account, else you will get "rate limit errors".


### Code formatting and style

To automatically sort the imports you can run
[`isort aintelope tests`](https://github.com/PyCQA/isort) from the root level of the project.
To autoformat python files you can use [`black .`](https://github.com/psf/black) from the root level of the project.
Configurations of the formatters can be found in `pyproject.toml`.
For linting/code style use [`flake8`](https://flake8.pycqa.org/en/latest/).

These tools can be invoked via `make`:

```bash
make isort
make format
make flake8
```


## Executing `aintelope`

In the folder `.vscode` there is a file named `launch.json.template`. Copy that file to `launch.json`. This is a VSCode launch configurations file, containing many launch configurations. (The original file named `launch.json.template` is necessary so that your local changes to launch configurations do not end up in the Git repository.)

Alternatively, try executing `make run-training-baseline`. You do not need VSCode for running this command. Then look in `aintelope/outputs`. This command will execute only one of many available launch configurations present in `launch.json`.

### Executing LLM agent

For LLM agent, there are the following launch configurations in `launch.json`:
- Run single environment with LLM agent and default params
- Run pipeline with LLM agent and default params
- Run BioBlue pipeline with LLM agent and default params
- Run multiple trials pipeline with LLM agent and default params
- Run multiple trials BioBlue pipeline with LLM agent and default params


## Actions map

The actions the agents can take have the following mapping:
```
  NOOP = 0
  LEFT = 1
  RIGHT = 2
  UP = 3
  DOWN = 4
```

Eating and drinking are not individual actions. Eating and drinking occurs always when an action ends with the agent being on top of a food or water tile, correspondingly. If the agent continues to stay on that tile then eating and drinking continues until the agent leaves. Likewise with collecting gold and silver. The agent is harmed by danger tile or predator, when the agent action ends up on a danger tile or predator tile. Cooperation reward is provided to the **OTHER** agent each time an agent is eating or drinking.

Additionally, when `observation_direction_mode` = 2 or `action_direction_mode` = 2 then the following actions become available:
```
  TURN_LEFT_90 = 5
  TURN_RIGHT_90 = 6
  TURN_LEFT_180 = 7
  TURN_RIGHT_180 = 8
```
By default, the separate turning actions are turned off.


## Human-playable demos

In the folder `aintelope\environments\demos\gridworlds\` are located the human-playable demo environments, which have same configuration as the benchmarks in our pipeline. Playing these human-playable demos manually can give you a better intuition of the rules and how the benchmarks work.

You can launch these Python files without additional arguments. 

You can move the agents around using arrow keys (left, right, up, down). For no-op action you can use space key. 

In food sharing environment there are two agents. In a human-playable demo these agents take turns. In an RL setting they agents take actions concurrently and the environment implements their actions in a random order (randomising the order for each turn).

The human-playable benchmark environments are in the following files:
```
food_unbounded.py
danger_tiles.py
predators.py
food_homeostasis.py
food_sustainability.py
food_drink_homeostasis.py
food_drink_homeostasis_gold.py
food_drink_homeostasis_gold_silver.py
food_sharing.py
```


## Windows

Aintelope code base is compatible with Windows. No extra steps needed. GPU computation works fine as well. WSL is not needed.


# -----

**Attribution & License**

This repository is a fork and derivative of "From homeostasis to resource sharing: Biologically and economically aligned multi-objective multi-agent gridworld-based AI safety benchmarks" (Roland Pihlakas et al.). Please cite the original work (DOI: `10.48550/arXiv.2410.00081`) — see `CITATION.cff` for citation metadata.  
Original upstream repository: https://github.com/biological-alignment-benchmarks/biological-alignment-gridworlds-benchmarks

License: see `LICENSE.txt`.  
Authors and contribution details: see `AUTHORS.md`.

