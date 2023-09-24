import os
import json
import yaml
import wandb
import logging
import numpy as np

from csv import writer
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter


class Logger():
    def __init__(self, cfg, run_name) -> None:

        self.cfg = cfg

        # Time run starts
        self.start_time = datetime.now()   

        # Config parameters
        self.log_name = os.path.join(cfg["logger"]["folder_path"], run_name)    
        self.enable_log_step = cfg["logger"]["log_step"]
        self.enable_log_episode = cfg["logger"]["log_episode"]
        self.enable_log_terminal = cfg["logger"]["log_terminal"]
       
        # Create log folder
        if not os.path.exists(self.log_name): os.makedirs(self.log_name)

        # Save run config
        config_file_path = os.path.join(self.log_name, 'config.json')  
        with open(config_file_path, 'w') as file: 
            json.dump(cfg, file, sort_keys=True, indent=4)

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

        # Set Files
        ## Episode File
        self.episode_file_path = os.path.join(self.log_name, 'episode.csv')  
        with open(self.episode_file_path, 'w') as file:
            episode_log = writer(file)
            episode_log.writerow(
                ["steps", "ep_steps", "episode", "reward", "mean_reward", "epsilon", "option_lengths"]
            )

        
    def print_config(self):

        parameters = yaml.dump(self.cfg)
        print("\nRUN: {}\n\nPARAMETERS:\n {}".format(self.log_name, parameters))

    def log_data(self, step, rewards, actor_loss, critic_loss, entropy, epsilon):

        if not self.enable_log_step:
            return

    def log_episode(self, steps, ep_steps, episode, reward, mean_reward, epsilon, option_lengths=None):

        if not self.enable_log_episode:
            return
        
        if self.enable_log_terminal:
            time = str(datetime.now()-self.start_time)
            logging.info(
                f"Episode: {episode:5d} | Steps: {steps:7d} | Reward={reward:.1f} "\
                f"| Episode Steps={ep_steps:5d} "\
                f"| Time={time:6s} | Epsilon={epsilon:.3f}")
            
        with open(self.episode_file_path, 'a') as file:
            episode_log = writer(file) 
            episode_log.writerow(
                [steps, ep_steps, episode, reward, mean_reward, epsilon, str(option_lengths)]
            )
        
    def close(self):
        pass
        


class TensorboardLogger(Logger):
    def __init__(self, cfg, run_name):
        super().__init__(cfg, run_name)

        self.tf_writer = None
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

        super().close()
        wandb.finish()