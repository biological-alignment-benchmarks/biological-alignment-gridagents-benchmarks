# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository:
# https://github.com/biological-alignment-benchmarks/biological-alignment-gridworlds-benchmarks

import logging
import os
import sys
import torch
import gc

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from aintelope.utils import disable_gym_warning, wait_for_enter

disable_gym_warning()

from aintelope.config.config_utils import (
    register_resolvers,
    get_score_dimensions,
    set_console_title,
)
from aintelope.experiments import run_experiment, run_experiment_with_retries

from aintelope.analytics import plotting, recording

logger = logging.getLogger("aintelope.__main__")


@hydra.main(version_base=None, config_path="config", config_name="config_experiment")
def aintelope_main(cfg: DictConfig) -> None:
    timestamp = str(cfg.timestamp)
    timestamp_pid_uuid = str(cfg.timestamp_pid_uuid)
    logger.info(f"timestamp: {timestamp}")
    logger.info(f"timestamp_pid_uuid: {timestamp_pid_uuid}")

    config_name = HydraConfig.get().job.config_name
    set_console_title(config_name + " : " + timestamp_pid_uuid)

    logger.info("Running training with the following configuration")
    logger.info(OmegaConf.to_yaml(cfg))
    score_dimensions = get_score_dimensions(cfg)

    # train
    (
        num_actual_train_episodes,
        training_run_was_terminated_early_due_to_nans,
        num_training_retries_used,
        _,
    ) = run_experiment_with_retries(
        cfg,
        experiment_name=cfg.experiment_name,
        score_dimensions=score_dimensions,
        test_mode=False,
        i_pipeline_cycle=0,
    )

    # test
    if (
        training_run_was_terminated_early_due_to_nans
        and cfg.hparams.model_params.skip_test_on_training_stop_on_nan_errors
    ):
        test_checkpoint_filenames = None
    else:
        (_, _, test_checkpoint_filenames) = run_experiment(
            cfg,
            experiment_name=cfg.experiment_name,
            score_dimensions=score_dimensions,
            test_mode=True,
            i_pipeline_cycle=0,
            num_actual_train_episodes=num_actual_train_episodes,
        )

        title = timestamp + " : " + cfg.experiment_name
        do_not_show_plot = cfg.hparams.unit_test_mode
        analytics(
            cfg,
            score_dimensions,
            title=title,
            experiment_name=cfg.experiment_name,
            num_actual_train_episodes=num_actual_train_episodes,
            training_run_was_terminated_early_due_to_nans=training_run_was_terminated_early_due_to_nans,
            test_checkpoint_filenames=test_checkpoint_filenames,
            do_not_show_plot=do_not_show_plot,
        )

        if not do_not_show_plot:
            # keep plots visible until the user decides to close the program
            wait_for_enter("Press [enter] to continue.")

    # / if training_run_was_terminated_early_due_to_nans and cfg.hparams.model_params.skip_test_on_training_stop_on_nan_errors:


# / def aintelope_main(cfg: DictConfig)


def analytics(
    cfg,
    score_dimensions,
    title,
    experiment_name,
    num_actual_train_episodes=-1,  # currently unused here
    training_run_was_terminated_early_due_to_nans=False,  # currently unused here
    test_checkpoint_filenames=None,  # currently unused here
    do_not_show_plot=False,
):
    # normalise slashes in paths. This is not mandatory, but will be cleaner to debug
    log_dir = os.path.normpath(cfg.log_dir)
    events_fname = cfg.events_fname
    num_train_episodes = cfg.hparams.num_episodes
    num_train_pipeline_cycles = cfg.hparams.num_pipeline_cycles

    savepath = os.path.join(log_dir, "plot_" + experiment_name + ".png")
    events = recording.read_events(log_dir, events_fname)

    torch.cuda.empty_cache()
    gc.collect()

    plotting.plot_performance(
        events,
        num_train_episodes,
        num_train_pipeline_cycles,
        score_dimensions,
        save_path=savepath,
        title=title,
        group_by_pipeline_cycle=False,
        do_not_show_plot=do_not_show_plot,
    )


if __name__ == "__main__":  # for multiprocessing support
    register_resolvers()
    aintelope_main()
