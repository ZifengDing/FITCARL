import torch
import json
import os
# import tqdm
import random
import numpy as np

class Trainer(object):
    def __init__(self, model, pg, optimizer, args):
        self.model = model
        self.pg = pg
        self.optimizer = optimizer
        self.args = args
        self.curr_ent_idx = 0

    def train_epoch(self, train_ent2quad, ntriple, old_support=None):
        self.model.train()
        total_loss = 0.0
        total_reward = 0.0
        # counter = 0

        self.num_unseen_ent = len(list(train_ent2quad.keys()))
        self.all_unseen_entities = list(train_ent2quad.keys())
        self.tasks = train_ent2quad
        [support, query], curr_unseen_ent = self.next_batch()
        support = torch.flatten(torch.tensor(support), start_dim=0, end_dim=1)
        query = torch.flatten(torch.tensor(query), start_dim=0, end_dim=1)
        src_sup = support[:,0]
        rel_sup = support[:,1]
        dst_sup = support[:,2]
        time_sup = support[:,3]
        src_que = query[:,0]
        rel_que = query[:,1]
        dst_que = query[:,2]
        time_que = query[:,3]
        if self.args.cuda:
            src_sup = src_sup.cuda()
            rel_sup = rel_sup.cuda()
            dst_sup = dst_sup.cuda()
            time_sup = time_sup.cuda()
            src_que = src_que.cuda()
            rel_que = rel_que.cuda()
            dst_que = dst_que.cuda()
            time_que = time_que.cuda()

        # adjust graph and state action space in meta-training
        if isinstance(old_support, np.ndarray): # if its after first time
            self.model.env.adjust_graph(old_support, support)
            self.model.env.remove_state_action_space_all(old_support)
            self.model.env.expand_state_action_space_all(support)
        else: # if it is the first time
            affected_back_ent = self.model.env.add_newfew(support)
            self.model.env.affected_back_ent = affected_back_ent
            self.model.env.expand_state_action_space_all(support)
        
        if self.args.sector_emb:
            self.model.update_sector_emb()
            self.model.update_ent_emb_sector()

        unseen_ent_emb = self.model.entity_learner(support, query, self.args)
        if self.args.sector_emb:
            unseen_ent_emb = self.model.update_ent_emb_sector_unseen(support, unseen_ent_emb)
        src_que_embeds = self.model.agent.ent_embs.forward_withraw(unseen_ent_emb, torch.zeros_like(time_que))
        dst_que_embs_static = self.model.agent.ent_embs.ent_embs(dst_que)

        all_loss, all_logits, _, current_entities, current_time, all_loss_reg, reward = self.model(src_que, src_que_embeds, time_que,
                                                                                                   rel_que, dst_que_embs_static, issupport=False)

        cum_discounted_reward = self.pg.calc_cum_discounted_reward_(reward)
        num_query = src_que.shape[0]
        reinfore_loss = self.pg.calc_reinforce_loss(all_loss, all_logits, cum_discounted_reward, all_loss_reg, num_query)
        self.pg.now_epoch += 1

        self.optimizer.zero_grad()
        reinfore_loss.backward()
        if self.args.clip_gradient:
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_gradient)
        self.optimizer.step()

        total_loss += reinfore_loss
        total_reward += torch.mean(cum_discounted_reward)

        return total_loss / self.num_unseen_ent, total_reward / self.num_unseen_ent, support

    def save_model(self, checkpoint_path='checkpoint.pth'):
        """Save the parameters of the model and the optimizer,"""
        argparse_dict = vars(self.args)
        with open(os.path.join(self.args.save_path, 'config.json'), 'w') as fjson:
            json.dump(argparse_dict, fjson)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()},
            os.path.join(self.args.save_path, checkpoint_path)
        )

    def next_one(self):
        if self.curr_ent_idx % self.num_unseen_ent == 0:
            random.shuffle(self.all_unseen_entities)
            self.curr_ent_idx = 0

        # get current relation and current candidates
        curr_unseen_ent = self.all_unseen_entities[self.curr_ent_idx]
        self.curr_ent_idx = (self.curr_ent_idx + 1) % self.num_unseen_ent  # shift current relation idx to next

        # get current tasks by curr_rel from all tasks and shuffle it
        curr_tasks = np.array(self.tasks[curr_unseen_ent])
        curr_tasks_idx = np.arange(0, len(curr_tasks), 1)
        curr_tasks_idx = np.random.choice(curr_tasks_idx, self.args.few + self.args.nq)
        support_triples = [curr_tasks[i] for i in curr_tasks_idx[:self.args.few]]    # [F, 4]
        query_triples = [curr_tasks[i] for i in curr_tasks_idx[self.args.few:]]

        return support_triples, query_triples, curr_unseen_ent

    def next_batch(self):
        next_batch_all = [self.next_one() for _ in range(self.num_unseen_ent)]
        support, query, curr_unseen_ent = zip(*next_batch_all)
        return [np.array(support), np.array(query)], curr_unseen_ent
    
    def to_head_mode(self, quads, unseen_ent, num_rel):
        new = quads.copy()
        mask = quads[:, 2] == unseen_ent
        flip_part = quads[mask][:, [2, 1, 0, 3]]
        flip_part[:, 1] += num_rel
        new[mask] = flip_part
        return new
