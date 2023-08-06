import os
import time
import logging
import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

import wandb


class TensorboardLogger():
    def __init__(self, logdir, run_name):
        self.log_name = logdir + '/' + run_name
        self.tf_writer = None
        self.start_time = time.time()
        self.n_eps = 0

        if not os.path.exists(self.log_name):
            os.makedirs(self.log_name)

        self.writer = SummaryWriter(self.log_name)

        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.log_name + '/logger.log'),
                ],
            datefmt='%Y/%m/%d %I:%M:%S %p'
            )

    def log_episode(self, steps, reward, mean_reward, option_lengths, ep_steps, epsilon):
        self.n_eps += 1
        logging.info(f"> ep {self.n_eps} done. total_steps={steps} | reward={reward} | episode_steps={ep_steps} "\
            f"| hours={(time.time()-self.start_time) / 60 / 60:.3f} | epsilon={epsilon:.3f}")
        self.writer.add_scalar(tag="episodic_rewards", scalar_value=reward, global_step=self.n_eps)
        self.writer.add_scalar(tag="episodic_mean_rewards", scalar_value=mean_reward, global_step=self.n_eps)
        self.writer.add_scalar(tag='episode_lengths', scalar_value=ep_steps, global_step=self.n_eps)

        # Keep track of options statistics
        for option, lens in option_lengths.items():
            # Need better statistics for this one, point average is terrible in this case
            self.writer.add_scalar(tag=f"option_{option}_avg_length", scalar_value=np.mean(lens) if len(lens)>0 else 0, global_step=self.n_eps)
            self.writer.add_scalar(tag=f"option_{option}_active", scalar_value=sum(lens)/ep_steps, global_step=self.n_eps)
    
    def log_data(self, step, rewards, actor_loss, critic_loss, entropy, epsilon, action, probs):

        if actor_loss:
            self.writer.add_scalar(tag="actor_loss", scalar_value=actor_loss.item(), global_step=step)
        if critic_loss:
            self.writer.add_scalar(tag="critic_loss", scalar_value=critic_loss.item(), global_step=step)
        self.writer.add_scalar(tag="policy_entropy", scalar_value=entropy, global_step=step)
        self.writer.add_scalar(tag="epsilon",scalar_value=epsilon, global_step=step)
        self.writer.add_scalar(tag="step_rewards",scalar_value=rewards, global_step=step)


class WanDBLogger():

    def __init__(self, wandb_project, run_name):
        self.start_time = time.time()


        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project=wandb_project,
            name=run_name,
            
            # track hyperparameters and run metadata
            # config=OmegaConf.to_container(cfc, resolve=True)
        )


        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(message)s',
            handlers=[
                logging.StreamHandler()
                ],
            datefmt='%Y/%m/%d %I:%M:%S %p'
            )
        
    def log_episode(self, steps, ep_steps, episode, reward, mean_reward, epsilon, option_lengths=None):
        logging.info(f"> ep {episode} done. total_steps={steps} | reward={reward} | episode_steps={ep_steps} "\
            f"| hours={(time.time()-self.start_time) / 60 / 60:.3f} | epsilon={epsilon:.3f}")

        wandb.log({"episode": episode, "reward": reward, "mean_reward": mean_reward, "steps": ep_steps, "epsilon":epsilon})

        if option_lengths:
            for option, lens in option_lengths.items():
                # Need better statistics for this one, point average is terrible in this case
                wandb.log({
                        f"option_{option}_avg_length": np.mean(lens) if len(lens)>0 else 0,
                        f"option_{option}_active": sum(lens)/ep_steps
                    })
                
    def log_data(self, step, rewards, actor_loss, critic_loss, entropy, epsilon, action, probs):

        if actor_loss:
            wandb.log({"actor_loss": actor_loss.item()})
        if critic_loss:
            wandb.log({"critic_loss": critic_loss.item()})

        wandb.log({"policy_entropy": entropy, "epsilon": epsilon, "step_rewards": rewards}) 


    def close(self):
        wandb.finish()