from collections import defaultdict
import logging
import numbers

import numpy as np
import torch as th


class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger
        self.use_tb = False
        self.use_sacred = False
        self.use_hdf = False
        self.wandb_run = None
        self.stats = defaultdict(list)

    def setup_tb(self, directory_name):
        from torch.utils.tensorboard import SummaryWriter
        self.tb_writer = SummaryWriter(directory_name)
        self.use_tb = True

    def setup_wandb(self, args, directory_name):
        """Explicit opt-in only; authentication comes from the user's environment."""
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("use_wandb=True requires installing wandb; use_wandb=False works offline") from exc
        self.wandb_run = wandb.init(
            config=vars(args), project=args.wandb_project,
            entity=args.wandb_entity, name=args.unique_token,
            dir=str(directory_name), mode=args.wandb_mode,
        )

    def setup_sacred(self, sacred_run_dict):
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def log_stat(self, key, value, t, to_sacred=True):
        if isinstance(value, th.Tensor):
            value = value.detach().item()
        elif isinstance(value, np.generic):
            value = value.item()
        if not isinstance(value, numbers.Number):
            raise TypeError("log_stat expects a scalar for {}".format(key))
        t = int(t)
        self.stats[key].append((t, value))
        # Only five entries are used for console summaries; Sacred is the history.
        del self.stats[key][:-5]
        if self.use_tb:
            self.tb_writer.add_scalar(key, value, t)
        if self.wandb_run is not None:
            self.wandb_run.log({key: value}, step=t)
        if self.use_sacred and to_sacred:
            self.sacred_info.setdefault("{}_T".format(key), []).append(t)
            self.sacred_info.setdefault(key, []).append(value)

    def print_recent_stats(self):
        if not self.stats.get("episode"):
            return
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(*self.stats["episode"][-1])
        index = 0
        for key, values in sorted(self.stats.items()):
            if key == "episode" or not values:
                continue
            index += 1
            window = 1 if key == "epsilon" else 5
            mean = np.mean([value for _, value in values[-window:]])
            log_str += "{:<25}{:>8.4f}".format(key + ":", mean)
            log_str += "\n" if index % 4 == 0 else "\t"
        self.console_logger.info(log_str)

    def close(self):
        if self.use_tb:
            self.tb_writer.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()


def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
