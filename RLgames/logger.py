import os
import time
import logging
import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

import wandb
import yaml


class Logger():
    def __init__(self, cfg, run_name) -> None:
        self.cfg = cfg
        self.log_name = os.path.join(cfg["logger"]["folder_path"], run_name)    
        self.enable_log_step = cfg["logger"]["log_step"]
        self.enable_log_episode = cfg["logger"]["log_episode"]
        self.enable_log_terminal = cfg["logger"]["log_terminal"]

        if not os.path.exists(self.log_name):
            os.makedirs(self.log_name)

        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.log_name + '/logger.log'),
                ],
            datefmt='%Y/%m/%d %I:%M:%S %p'
        )

        self.print_config()

        
    def print_config(self):
        parameters = yaml.dump(self.cfg)
        print("\nRUN: {}\n\nPARAMETERS:\n {}".format(self.log_name, parameters))

    def log_episode(self, steps, ep_steps, episode, reward, mean_reward, epsilon, option_lengths=None):
        if not self.enable_log_episode:
            return
        
        if self.enable_log_terminal:
            logging.info(f"> ep {episode} done. total_steps={steps} | reward={reward:5d} | episode_steps={ep_steps:7d} "\
                f"| hours={(time.time()-self.start_time) / 60 / 60:.3f} | epsilon={epsilon:.3f}")
        
    def close(self):
        pass


class TensorboardLogger(Logger):
    def __init__(self, cfg, run_name):
        super().__init__(cfg, run_name)

        self.tf_writer = None
        self.start_time = time.time()
        self.writer = SummaryWriter(self.log_name)
        
    def log_episode(self, steps, ep_steps, episode, reward, mean_reward, epsilon, option_lengths=None):
        if not self.enable_log_episode:
            return
        super().log_episode(steps, ep_steps, episode, reward, mean_reward, epsilon, option_lengths)

        self.writer.add_scalar(tag="episodic_rewards", scalar_value=reward, global_step=episode)
        self.writer.add_scalar(tag="episodic_mean_rewards", scalar_value=mean_reward, global_step=episode)
        self.writer.add_scalar(tag='episode_lengths', scalar_value=ep_steps, global_step=episode)

        # Keep track of options statistics
        if option_lengths:
            for option, lens in option_lengths.items():
                # Need better statistics for this one, point average is terrible in this case
                self.writer.add_scalar(tag=f"option_{option}_avg_length", scalar_value=np.mean(lens) if len(lens)>0 else 0, global_step=episode)
                self.writer.add_scalar(tag=f"option_{option}_active", scalar_value=sum(lens)/ep_steps, global_step=episode)
    
    def log_data(self, step, rewards, actor_loss, critic_loss, entropy, epsilon):
        if not self.enable_log_step:
            return

        if actor_loss:
            self.writer.add_scalar(tag="actor_loss", scalar_value=actor_loss.item(), global_step=step)
        if critic_loss:
            self.writer.add_scalar(tag="critic_loss", scalar_value=critic_loss.item(), global_step=step)
        self.writer.add_scalar(tag="policy_entropy", scalar_value=entropy, global_step=step)
        self.writer.add_scalar(tag="epsilon",scalar_value=epsilon, global_step=step)
        self.writer.add_scalar(tag="step_rewards",scalar_value=rewards, global_step=step)


class WanDBLogger(Logger):

    def __init__(self, cfg, run_name):
        super().__init__(cfg, run_name)

        self.start_time = time.time()

        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project=cfg["logger"]["wandb_project"],
            name=run_name,
            # track hyperparameters and run metadata
            config=cfg
        )      
    
        
    def log_episode(self, steps, ep_steps, episode, reward, mean_reward, epsilon, option_lengths=None):
        if not self.enable_log_episode:
            return
        
        super().log_episode(steps, ep_steps, episode, reward, mean_reward, epsilon, option_lengths)

        wandb.log({
            "episode": episode, "reward": reward, 
            "mean_reward": mean_reward, "steps": ep_steps, "epsilon":epsilon})

        if option_lengths:
            for option, lens in option_lengths.items():
                # Need better statistics for this one, point average is terrible in this case
                wandb.log({
                        f"option_{option}_avg_length": np.mean(lens) if len(lens)>0 else 0,
                        f"option_{option}_active": sum(lens)/ep_steps
                    })
                
    def log_data(self, step, rewards, actor_loss, critic_loss, entropy, epsilon):
        if not self.enable_log_step:
            return

        if actor_loss:
            wandb.log({"actor_loss": actor_loss.item()})
        if critic_loss:
            wandb.log({"critic_loss": critic_loss.item()})

        wandb.log({"step": step, "policy_entropy": entropy, "epsilon": epsilon, "step_rewards": rewards}) 


    def close(self):
        wandb.finish()