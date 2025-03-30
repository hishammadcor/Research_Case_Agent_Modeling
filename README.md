# 🧠 Project Title: **LLM Persona-Based Survey Simulation and Evaluation**

## 📘 Introduction

This repository contains a modular Python framework for evaluating how large language models (LLMs) respond to survey questions when prompted as specific personas. The core idea is to simulate group-based responses by prompting the LLM with persona-specific instructions (e.g., “Christian Protestant,” “Jewish White with Bachelor”) and compare the results against real survey data.

Key components of the system include:

- 🧩 **Prompt generation** from structured CSV survey definitions.
- 🧠 **Persona injection** using customized JSON configurations.
- 🛠️ **LLM API integration** for automated response generation.
- 📊 **Evaluation toolkit** to analyze differences across groups using statistical measures like Chi-Square, JS Divergence, Spearman/Kendall correlations, and visualizations.

This setup enables detailed insights into how aligned model outputs are with real human data, especially across diverse sociocultural and demographic groups.


## Development Environment Preparation

### Pyenv

Managing multiple version of a python on a single machine is alot easier via `pyenv` 

- **linux/mac** users could follow the installation steps from the [link](https://github.com/pyenv/pyenv)

@note: please make sure, you have the build dependencies

```
apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

then enable `pyenv` on the terminal via

```
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
```

- **windows** users could follow the installation steps from the [link](https://github.com/pyenv-win/pyenv-win)

@note: Windows installations some times does not update the path until the restart of a pc. If the `pyenv` executable is not available, restart might be required.

Once the `pyenv` is install you can install any version of the python via

```
pyenv install x.y.z
```

then make it globally or locally available. Our suggetion is the the selection of a global version should not be changed frequently. Local enablemant of the any installed python version is possible via `pyenv local x.y.z`. That create a `.python-version` file in that location, and next time the shell activations will follow the set python version unless you update/remove the file.

At this point select and install a python version globally first e.g. `pyenv install 3.12` and then `pyenv global 3.12`


### Pipx 

`Pipx` is an isolated installation tool for indipentent applications. It enables clean global environments for python versions. Any dependency installed via `pipx` will not intefere with your main environment therefore remove/installing any other tool will be safer.

You can install `pipx` right after updating your `pip`

```
pip install --upgrade pip
pip install pipx
```

at this stage it will be available on your shell.

### Poetry

This project is using `poetry` as a dependency management, build, and documentation generation tool. `poetry` can be installed via `pipx`. This project is using the version `1.8.3` at the moment 

```
pipx install poetry
```

At this point all the required tools are installed. You can check the current required python version from `.python-version` and install it via `pyenv`. e.g `pyenv install 3.8.10`.

`poetry` has certaion default configurations, to make it easy to use jupyter notebooks and vscode virtual environment shell enablement, we are recomending to set the following two configuration options

- To use the .python-version option that is located at the root of the project. Enable 
`poetry config virtualenvs.prefer-active-python true`

- To have the virtual environment at the root of the project in a `.venv` directory. Enable
`poetry config virtualenvs.in-project true`

You can revert any of the chages by turning the `true` flag to `false`, and current configuration list can be seen via `poetry config --list`

After that, you run this command to setup the Environment after cloning the repository

```
poetry install
```

Then you can activate the environment by running

```
poetry shell
```

If you want to install a library or a dependency you must use this command

```
poetry add LIBRARY
```
e.g ` poetry add pandas`. This makes sure that the version and its dependencies are compatible with the already installed libraries and dependencies

