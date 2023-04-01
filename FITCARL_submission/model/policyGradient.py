import torch
import numpy as np
import math
from model.baseline import ReactiveBaseline

class PG(object):
    def __init__(self, config):
        self.config = config
        self.positive_reward = 1.0
        self.negative_reward = 0.0
        self.baseline = ReactiveBaseline(config, config['lambda'])
        self.now_epoch = 0

    def get_reward(self, current_entites, answers):
         positive = torch.ones_like(current_entites, dtype=torch.float32) * self.positive_reward
         negative = torch.ones_like(current_entites, dtype=torch.float32) * self.negative_reward
         reward = torch.where(current_entites == answers, positive, negative)
         return reward

    def get_reward_continuous(self, current_ent_embs, answer_embs):
        reward = torch.sigmoid(5-torch.norm(current_ent_embs - answer_embs, p=2, dim=1))
        return reward

    def calc_cum_discounted_reward(self, rewards):
        running_add = torch.zeros([rewards.shape[0]])
        cum_disc_reward = torch.zeros([rewards.shape[0], self.config['path_length']])
        if self.config['cuda']:
            running_add = running_add.cuda()
            cum_disc_reward = cum_disc_reward.cuda()

        cum_disc_reward[:, self.config['path_length'] - 1] = rewards
        for t in reversed(range(self.config['path_length'])):
            running_add = self.config['gamma'] * running_add + cum_disc_reward[:, t]
            cum_disc_reward[:, t] = running_add

        return cum_disc_reward

    def calc_cum_discounted_reward_(self, rewards):
        running_add = torch.zeros([rewards[0].shape[0]])
        cum_disc_reward = torch.cat(rewards, dim=1)
        if self.config['cuda']:
            running_add = running_add.cuda()
            cum_disc_reward = cum_disc_reward.cuda()

        # cum_disc_reward[:, self.config['path_length'] - 1] = rewards
        for t in reversed(range(self.config['path_length'])):
            running_add = self.config['gamma'] * running_add + cum_disc_reward[:, t]
            cum_disc_reward[:, t] = running_add

        return cum_disc_reward

    def entropy_reg_loss(self, all_logits):
        all_logits = torch.stack(all_logits, dim=2)
        entropy_loss = - torch.mean(torch.sum(torch.mul(torch.exp(all_logits), all_logits), dim=1))
        return entropy_loss

    def calc_reinforce_loss(self, all_loss, all_logits, cum_discounted_reward, all_loss_reg, num_query):
        loss = torch.stack(all_loss, dim=1)
        loss_reg = torch.stack(all_loss_reg, dim=1)
        base_value = self.baseline.get_baseline_value()
        final_reward = cum_discounted_reward - base_value

        reward_mean = torch.mean(final_reward)
        reward_std = torch.std(final_reward) + 1e-6
        final_reward = torch.div(final_reward - reward_mean, reward_std)

        loss = torch.mul(loss, final_reward)
        loss_reg = torch.sum(loss_reg) / num_query
        entropy_loss = self.config['ita'] * math.pow(self.config['zita'], self.now_epoch) * self.entropy_reg_loss(all_logits)

        total_loss = torch.mean(loss) - entropy_loss + 0.000000001 * loss_reg

        return total_loss
