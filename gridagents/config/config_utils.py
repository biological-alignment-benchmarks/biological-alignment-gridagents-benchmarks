# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository:
# https://github.com/biological-alignment-benchmarks/biological-alignment-gridagents-benchmarks

import copy
from ast import literal_eval
from pathlib import Path
import zipfile
import os
import sys
import uuid
import time

# import torch  # comment-out: optimisation: load torch lazyly only when needed. This speeds up analytics, for example.

from omegaconf import DictConfig, OmegaConf


def set_console_title(title):
    # see also https://en.wikipedia.org/wiki/ANSI_escape_code#OSC
    # and https://gist.github.com/fdncred/c649b8ab3577a0e2873a8f229730e939#supported-oscs
    try:
        if os.name == "nt":
            import ctypes

            ctypes.windll.kernel32.SetConsoleTitleW(
                title
            )  # https://stackoverflow.com/questions/7387276/set-windows-command-line-terminal-title-in-python
        else:
            term = os.getenv("TERM")
            if (
                term[:5] == "xterm" or term == "vt100"
            ):  # xterm may have various suffices in after "xterm"
                cmd = "\x1B]0;{}\x07".format(title)
                print(
                    cmd
                )  # https://www.thecodingforums.com/threads/changing-the-terminal-title-bar-with-python.965355/
            elif os.name == "posix":  # linux or mac
                import platform

                system = platform.system()
                if (
                    system == "Linux"
                ):  # TODO: does this OSC 2 command here work in cases when OSC 0 does not work? If they both do not work in same cases, then there is no need for separate xterm, Linux, and Darwin branches, they could be all merged into one
                    cmd = "\x1B]2;{}\x07".format(
                        title
                    )  # https://bbs.archlinux.org/viewtopic.php?id=85567
                    sys.stdout.write(cmd)
                elif system == "Darwin":  # same as xterm
                    # Since Ubuntu 16.04, escape sequence used is invalid due to $PS1 environment variable. Python interpreter process can not modify it (a kind of system process setting). So the only way is to change $PS1 in ~/.bashrc. - https://github.com/grimmer0125/terminal-title-change/wiki
                    # Though it works fine in 24.04.1 LTS
                    cmd = "\x1B]0;{}\x07".format(
                        title
                    )  # https://github.com/grimmer0125/terminal-title-change/blob/master/termtitle/termtitle.py
                    sys.stdout.write(cmd)
    except Exception:
        pass


def get_project_path(path_from_root: str) -> Path:
    project_root = Path(__file__).parents[2]
    return project_root / path_from_root


def custom_now(format: str = "%Y%m%d%H%M%S") -> str:
    return time.strftime(format)


def append_pid_and_uuid(timestamp: str) -> str:
    pid = os.getpid()
    unique_id = (
        uuid.uuid1()
    )  # Generate a UUID from a host ID, sequence number, and the current time. If node is not given, getnode() is used to obtain the hardware address. If clock_seq is given, it is used as the sequence number; otherwise a random 14-bit sequence number is chosen.  # TODO: uuid1 is good for multi-machine cloud computing. Other uuid methods might be preferrable in case of privacy considerations. See https://docs.python.org/3/library/uuid.html
    result = f"{timestamp}_{pid}_{unique_id}"
    return result


def create_range(start, exclusive_end):
    return list(range(start, exclusive_end))


def minus_3(entry):
    if entry is None:
        return None
    elif hasattr(entry, "__iter__"):  # isinstance(entry, list) does not work here
        return [x - 3 for x in entry]
    else:
        return entry - 3


def muldiv(entry, multiplier, divisor):
    if entry is None:
        return None
    elif hasattr(entry, "__iter__"):  # isinstance(entry, list) does not work here
        return [int(x * multiplier / divisor) for x in entry]
    else:
        return int(entry * multiplier / divisor)


def register_resolvers() -> None:
    OmegaConf.register_new_resolver("custom_now", custom_now)
    OmegaConf.register_new_resolver("abs_path", get_project_path)
    OmegaConf.register_new_resolver(
        "append_pid_and_uuid", append_pid_and_uuid, use_cache=True
    )  # NB! need to enable caching else the pid_and_uuid will change at random moments during execution, leading to errors
    OmegaConf.register_new_resolver("minus_3", minus_3)
    OmegaConf.register_new_resolver("muldiv", muldiv)
    OmegaConf.register_new_resolver("range", create_range)


def get_score_dimensions(cfg: DictConfig):
    scores = cfg.hparams.env_params.scores
    dimensions = set()
    for event_name, score_dims_dict in scores.items():
        score_dims_dict = literal_eval(score_dims_dict)
        for dimension, value in score_dims_dict.items():
            if value != 0:  # ignore zero valued score dimensions
                dimensions.add(dimension)
    dimensions = list(dimensions)
    dimensions.sort()
    return dimensions


def get_pipeline_score_dimensions(cfg: DictConfig, pipeline_config: DictConfig):
    dimensions = set()
    for env_conf in pipeline_config:
        experiment_cfg = copy.deepcopy(
            cfg
        )  # need to deepcopy in order to not accumulate keys that were present in previous experiment and are not present in next experiment
        OmegaConf.update(  # need to merge configs here too since dimensions inside scores are not merged, but instead overwritten by experiment config. If main config has some score dimension that experiment does not have, then then that score dimension should not be used
            experiment_cfg, "hparams", pipeline_config[env_conf], force_add=True
        )
        experiment_score_dimensions = get_score_dimensions(experiment_cfg)
        for dimension in experiment_score_dimensions:
            dimensions.add(dimension)

    dimensions = list(dimensions)
    dimensions.sort()
    return dimensions


def set_priorities():
    """Sets CPU priorities in order to avoid slowing down the system"""

    try:
        import psutil

        if hasattr(psutil, "Process"):
            pid = os.getpid()

            p = psutil.Process(pid)

            # set to lowest  priority, this is Windows only, on Unix use ps.nice(19)
            # On UNIX this is a number which usually goes from -20 to 20. The higher the nice value, the lower the priority of the process.
            # https://psutil.readthedocs.io/en/latest/#psutil.Process.nice
            p.nice(
                psutil.IDLE_PRIORITY_CLASS if os.name == "nt" else 20
            )  # TODO: config

            # On Windows only *ioclass* is used and it can be set to 2
            # (normal), 1 (low) or 0 (very low).
            p.ionice(0 if os.name == "nt" else psutil.IOPRIO_CLASS_IDLE)

            # print("Priorities set...")

    except Exception as msg:
        print("run pip install psutil")

    if os.name == "nt":
        try:  # psutil fails to set IO priority under Windows for some reason
            import win32process  # TODO: use ctypes.windll.kernel32 instead?

            win32process.SetThreadPriority(
                -2, -15
            )  # NB! -2: win32api.GetCurrentThread()  # -15: Idle priority, is lower than THREAD_MODE_BACKGROUND_BEGIN
            win32process.SetThreadPriorityBoost(-2, False)

            # NB! do not call win32process.SetPriorityClass(-1, 0x00100000) (PROCESS_MODE_BACKGROUND_BEGIN) since that would significantly reduce GPU load. Also, it would also cause constant page faults and swap disk writes.

            win32process.SetProcessPriorityBoost(-1, False)

        except Exception as msg:
            print("run pip install pywin32")


def set_memory_limits():
    """Sets memory usage limits in order to avoid crashing the system"""

    # TODO: read limits from config
    if os.name == "nt":
        mem_limit = 40 * 1024 * 1024 * 1024
        min_free_swap = 5 * 1024 * 1024 * 1024

        from gridagents.config.windows_jobobject import set_mem_commit_limit

        try:
            set_mem_commit_limit(os.getpid(), mem_limit, min_free_swap)
        except Exception as msg:
            print("run pip install psutil")
    else:  # / if os.name == 'nt':
        data_size_limit = 40 * 1024 * 1024 * 1024
        address_space_size_limit = 400 * 1024 * 1024 * 1024

        from gridagents.config.linux_rlimit import set_mem_limits

        set_mem_limits(data_size_limit, address_space_size_limit)


def select_gpu(gpu_index=None):
    if gpu_index is None:
        gpu_index = os.environ.get("GRIDAGENTS_GPU")

    # TODO: run some threads on CPU if the available GPU-s do not support the required amount of threads

    if gpu_index is not None:
        import torch  # Optimisation: load torch lazyly only when needed. This speeds up analytics, for example.

        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            print(
                "No CUDA GPU available, ignoring assigned GPU index, will be using CPU as a CUDA device"
            )
            return

        gpu_index = int(gpu_index)
        torch.cuda.set_device(gpu_index)
        device_name = torch.cuda.get_device_name(gpu_index)
        print(f"Using CUDA GPU {gpu_index} : {device_name}")
    else:
        # for each next experiment select next available GPU to maximally balance the load considering multiple running processes
        rotate_active_gpu_selection()


def rotate_active_gpu_selection():
    """Rotates over available GPU-s, selecting a next GPU for each subsequent
    program launch in order to distribute the workload of concurrently running
    programs over all available GPU-s.

    If you want to restrict the set of GPU-s the program selects from, then
    specify the CUDA_VISIBLE_DEVICES environment variable before starting the
    program like this:

    In Linux:
    export CUDA_VISIBLE_DEVICES=3,4

    In Windows:
    set CUDA_VISIBLE_DEVICES=3,4

    In VSCode, modify launch.json:
    "env": {
        "CUDA_VISIBLE_DEVICES": "3,4"
    }

    where 3,4 are the device numbers you want to enable for the program to choose from.
    The device numbers start from 0.
    """

    import torch  # Optimisation: load torch lazyly only when needed. This speeds up analytics, for example.

    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        print("No CUDA GPU available, will be using CPU as a CUDA device")
        return

    elif gpu_count == 1:
        # Even if only one device is enabled for CUDA, it might be not the first one by number
        # if it is set by CUDA_VISIBLE_DEVICES environment variable. Therefore, lets log it.
        gpu_counter = torch.cuda.current_device()
        print("Using the only available CUDA GPU")

    else:
        import sqlite3

        conn = sqlite3.connect("gpu_counter2.db")

        # NB! dummy column is needed to ensure that the INSERT is done only during first call and not on subsequent calls. There must be only one row in the table.
        cmd = """
        CREATE TABLE IF NOT EXISTS gpu_counter_table 
        (dummy INTEGER UNIQUE, gpu_counter INTEGER);
        INSERT OR IGNORE INTO gpu_counter_table (dummy, gpu_counter) VALUES (0, 0);
        """
        cursor = conn.cursor()
        cursor.executescript(cmd)
        conn.commit()

        cursor = conn.cursor()
        cursor.execute("BEGIN EXCLUSIVE TRANSACTION")
        cursor.execute("UPDATE gpu_counter_table SET gpu_counter = gpu_counter + 1")
        cursor = cursor.execute("SELECT gpu_counter FROM gpu_counter_table")
        gpu_counter = cursor.fetchone()[0]
        conn.commit()

        conn.close()

        gpu_counter = gpu_counter % gpu_count

        torch.cuda.set_device(gpu_counter)

    device_name = torch.cuda.get_device_name(gpu_counter)
    print(f"Using CUDA GPU {gpu_counter} : {device_name}")
    return


def get_project_root():
    project_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")
    )
    return project_root


def archive_code(cfg):
    """Archives the current version of the program to log folder"""

    # archive installed module versions into a txt file
    save_pip_list(cfg)

    project_root = get_project_root()
    zip_path = os.path.join(
        os.path.normpath(cfg.log_dir), "gridagents_code_archive.zip"
    )
    archive_code_in_dir([project_root], zip_path)

    # TODO: alternatively, import the module and check its path?
    code_directory_paths = (
        [  # the dependency is installed in different locations depending on setup
            os.path.join(project_root, "ai_safety_gridworlds"),
            os.path.join(project_root, "ai-safety-gridworlds"),
            os.path.join(project_root, "src", "ai_safety_gridworlds"),
            os.path.join(project_root, "src", "ai-safety-gridworlds"),
            os.path.join(
                project_root, "venv_gridagents", "src", "ai_safety_gridworlds"
            ),
            os.path.join(
                project_root, "venv_gridagents", "src", "ai-safety-gridworlds"
            ),
        ]
    )
    zip_path = os.path.join(
        os.path.normpath(cfg.log_dir), "gridworlds_code_archive.zip"
    )
    archive_code_in_dir(code_directory_paths, zip_path)

    # TODO: alternatively, import the module and check its path?
    code_directory_paths = (
        [  # the dependency is installed in different locations depending on setup
            os.path.join(project_root, "zoo_to_gym_multiagent_adapter"),
            os.path.join(project_root, "zoo-to-gym-multiagent-adapter"),
            os.path.join(project_root, "src", "zoo_to_gym_multiagent_adapter"),
            os.path.join(project_root, "src", "zoo-to-gym-multiagent-adapter"),
            os.path.join(
                project_root,
                "venv_gridagents",
                "src",
                "zoo_to_gym_multiagent_adapter",
            ),
            os.path.join(
                project_root,
                "venv_gridagents",
                "src",
                "zoo-to-gym-multiagent-adapter",
            ),
        ]
    )
    zip_path = os.path.join(
        os.path.normpath(cfg.log_dir), "multiagent_adapter_code_archive.zip"
    )
    archive_code_in_dir(code_directory_paths, zip_path)


def save_pip_list(cfg):
    """Creates a file containing list of installed Python packages and their exact versions."""

    from importlib import metadata

    packages = [f"{dist.name}=={dist.version}" for dist in metadata.distributions()]
    packages = sorted(packages, key=str.casefold)
    packages = "\n".join(packages)

    filename = os.path.join(cfg.log_dir, "package_versions.txt")
    with open(filename, "w", 1024 * 1024, encoding="utf-8") as fh:
        fh.write(packages)
        fh.flush()


def is_junction(path: str) -> bool:
    if os.name != "nt":
        return False
    path = Path(
        os.path.normpath(path)
    )  # normpath: remove ".." parts, else path.absolute() != path.resolve() would be True for normal folders as well
    return path.absolute() != path.resolve()


def archive_code_in_dir(directory_paths, zip_path):
    for directory_path in directory_paths:
        if os.path.exists(directory_path):
            directory_path_is_junction = is_junction(directory_path)
            with zipfile.ZipFile(zip_path, "w") as ziph:
                for root, dirs, files in os.walk(
                    directory_path, topdown=True, followlinks=False
                ):
                    # When topdown is True, the caller can modify the dirnames list in-place (perhaps using del or slice assignment), and walk() will only recurse into the subdirectories whose names remain in dirnames; this can be used to prune the search
                    # https://docs.python.org/3/library/os.html#os.walk
                    dirs_to_skip = []
                    for dir_ in dirs:
                        if (
                            dir_[:1]
                            == "."  # ignore dirs that start with dot (.git, .vshistory, etc), also ignore outputs folder
                            or dir_
                            in [
                                ".git",
                                "__pycache__",
                                "gridsearch_cache",
                                "outputs",
                                "screenshots",
                                "src",
                                "venv_gridagents",
                            ]
                            or (
                                not directory_path_is_junction  # if directory_path (for example, ai_safety_gridworlds) is junction then do not exclude its subfolders
                                and is_junction(
                                    os.path.join(root, dir_)
                                )  # os.walk follows Windows junctions even when followlinks=False but we want to exclude them
                            )
                        ):
                            dirs_to_skip.append(
                                dir_
                            )  # cannot remove dir directly from the list that is being iterated, else following dirs may be skipped from check
                    for dir_ in dirs_to_skip:
                        dirs.remove(dir_)

                    for file in files:
                        extension = os.path.splitext(file)[1]
                        if extension in [
                            ".cff",
                            ".gitignore",
                            ".ipynb",
                            ".md",
                            ".py",
                            ".template",
                            ".toml",
                            ".txt",
                            ".yaml",
                        ] or file in ["AUTHORS", "LICENSE", "launch.json"]:
                            ziph.write(
                                os.path.join(root, file),
                                os.path.relpath(
                                    os.path.join(root, file),
                                    os.path.join(directory_path, ".."),
                                ),
                            )
            break
        # / if os.path.exists(directory_path):
    # / for directory_path in directory_paths:


# used for disabling context objects like multiprocessing pool or progressbar
class DummyContext(object):
    def __init__(self, *args, **kwargs):
        pass

    # context manager functionality requires this method to be explicitly implemented
    def __enter__(self):
        return self

    # context manager functionality requires this method to be explicitly implemented
    def __exit__(self, type, value, traceback):
        return

    def _blackHoleMethod(*args, **kwargs):
        return

    def __getattr__(self, attr):
        return self._blackHoleMethod


# / class DummyContext(object):


def read_code_version():
    project_root = get_project_root()

    pyproject_toml = os.path.join(project_root, "pyproject.toml")
    if os.path.exists(pyproject_toml):
        import toml

        toml_data = toml.load(pyproject_toml)
        toml_version = str(toml_data.get("project", {}).get("version", "0.0.0.0"))
    else:
        toml_version = "0.0.0.0"

    citation_cff = os.path.join(project_root, "CITATION.cff")
    if os.path.exists(citation_cff):
        import yaml

        with open(citation_cff, "r", encoding="utf-8") as fh:
            yaml_data = yaml.safe_load(fh)
        cff_version = str(yaml_data.get("version", "0.0.0.0"))
    else:
        cff_version = "0.0.0.0"

    from packaging.version import parse as parse_version

    toml_version = parse_version(toml_version)
    cff_version = parse_version(cff_version)
    result = str(
        max(toml_version, cff_version)
    )  # take latest version in case one of the files is forgotten to be updated

    return result


# / def read_code_version():

code_version = (
    read_code_version()
)  # read code version at the start of the program so that if the files are updated while program is running then that will not mess up the version info


def get_code_version():
    return code_version


def get_gpu_name():
    import torch

    result = torch.cuda.get_device_name() if torch.cuda.device_count() > 0 else "CPU"
    return result


def get_computer_name():
    import platform

    return platform.node()
