import torch
import torch.nn as nn
import numpy as np
from .modules import TransformerEncoder


class Episode(nn.Module):
    def __init__(self, env, agent, config):
        super(Episode, self).__init__()
        self.config = config
        self.env = env
        self.agent = agent
        self.path_length = config['path_length']
        self.num_rel = config['num_rel']
        self.max_action_num = config['max_action_num']
        self.entity_learning_layer = nn.Linear(2 * (config['ent_dim'] - config['time_dim']),
                                               config['ent_dim'] - config['time_dim'])
        self.proj1 = nn.Linear(config['ent_dim'] - config['time_dim'], config['ent_dim'] - config['time_dim'])
        self.proj2 = nn.Linear(config['ent_dim'] - config['time_dim'], config['ent_dim'] - config['time_dim'])
        if self.config['support_learner'] == 'transformer':
            self.transformer = TransformerEncoder(self.agent.ent_embs, time_dim=config['time_dim'],
                                                  model_dim=config['ent_dim'] - config['time_dim'], num_heads=2,
                                                  num_layers=2,
                                                  max_seq_len=config['few'] + 1, with_pos=False)
        if self.config['sector_emb']:
            self.sec_emb = nn.Embedding(self.config['num_ent'], config['ent_dim'] - config['time_dim'])
            self.ent2sec = None
            self.back_ent = None
            self.ent2sec_weight = torch.nn.Parameter(
                torch.tensor(np.random.uniform(-1, 1, (config['ent_dim'] - config['time_dim'],
                                                       config['ent_dim'] - config['time_dim'])), dtype=torch.float,
                             requires_grad=True))
            self.sec2ent_weight = torch.nn.Parameter(
                torch.tensor(np.random.uniform(-1, 1, (config['ent_dim'] - config['time_dim'],
                                                       config['ent_dim'] - config['time_dim'])), dtype=torch.float,
                             requires_grad=True))

    def update_sector_emb(self):
        for i in range(self.ent2sec.shape[1] - 1):  # last sector is for padding
            ents4sec = torch.nonzero(self.ent2sec.transpose(1, 0)[i, :])
            ents4sec_fil = []
            for ent in ents4sec:
                if int(ent.item()) not in self.back_ent:
                    continue
                ents4sec_fil.append(ent)
            if len(ents4sec_fil) == 0:
                continue
            ents4sec_fil = torch.tensor(ents4sec_fil)
            if self.config['cuda']:
                ents4sec_fil = ents4sec_fil.cuda()
            ents_emb4sec = self.agent.ent_embs.ent_embs(ents4sec_fil)
            self.sec_emb.weight.data[i, :].copy_(torch.mean(torch.matmul(ents_emb4sec, self.ent2sec_weight), dim=0))

    def update_ent_emb_sector(self):
        for i in range(self.config['ent_dim'] - 1):  # last entity is for padding
            secs4ent = torch.nonzero(self.ent2sec[i, :]).squeeze(1)
            if self.config['cuda']:
                secs4ent = secs4ent.cuda()
            secs_emb4ent = self.sec_emb(secs4ent)
            if int(i) in self.back_ent:  # back entity do not initialize with sector
                continue
            else:
                self.agent.ent_embs.ent_embs.weight.data[i, :] = 0.000001 * self.agent.ent_embs.ent_embs_copy.weight.data[i,
                                                                            :] + \
                                                                 (torch.mean(
                                                                     torch.matmul(secs_emb4ent, self.sec2ent_weight),
                                                                     dim=0))

    def update_ent_emb_sector_unseen(self, support, unseen_ent_emb):
        unseen_ent = support[:, 0]
        if self.config['cuda']:
            unseen_ent.cuda()
        for i, uent in enumerate(unseen_ent):
            secs4ent = torch.nonzero(self.ent2sec[i, :]).squeeze(1)
            if self.config['cuda']:
                secs4ent = secs4ent.cuda()
            secs_emb4ent = self.sec_emb(secs4ent)
            unseen_ent_emb[i, :] += 0.000001 * (torch.mean(torch.matmul(secs_emb4ent, self.sec2ent_weight), dim=0))
        return unseen_ent_emb

    def load_pretrain(self, file1, file2):
        # load entity emb
        pretrain_ent_emb = np.load(file1)
        self.agent.ent_embs.ent_embs.weight.data[:-1, :].copy_(torch.from_numpy(pretrain_ent_emb))
        if self.config['sector_emb']:
            self.agent.ent_embs.ent_embs_copy.weight.data[:-1, :].copy_(torch.from_numpy(pretrain_ent_emb))

        # load relation emb
        pretrain_rel_emb = np.load(file2)
        pretrain_rel_emb_noinv, pretrain_rel_emb_inv = pretrain_rel_emb[:self.num_rel, :], pretrain_rel_emb[
                                                                                           self.num_rel:, :]
        self.agent.rel_embs.weight.data[:self.num_rel, :].copy_(torch.from_numpy(pretrain_rel_emb_noinv))
        self.agent.rel_embs.weight.data[self.num_rel + 1:self.num_rel * 2 + 1, :].copy_(
            torch.from_numpy(pretrain_rel_emb_inv))

    def entity_learner(self, support, query, args):
        current_uent = None
        # same_unseen_track = []
        k = 0
        multi_shot_support_for_query = []
        if args.cuda:
            unseen_ent = support[:, 0].cuda()
            # print(unseen_ent)
            sup_rel = support[:, 1].cuda()
            sup_obj = support[:, 2].cuda()
            sup_time = support[:, 3].cuda()
        else:
            unseen_ent = support[:, 0]
            # print(unseen_ent)
            sup_rel = support[:, 1]
            sup_obj = support[:, 2]
            sup_time = support[:, 3]
        mapped_unseen_ent_embs = self.entity_map(unseen_ent, sup_rel, sup_obj, sup_time, model=self.config['entity_learner'])

        for i, uent in enumerate(unseen_ent):
            if uent != current_uent:
                sup_time_cur = support[:, 3][k * args.few:(k + 1) * args.few]
                if args.cuda:
                    que = query[k * args.nq:(k + 1) * args.nq, :].cuda()
                else:
                    que = query[k * args.nq:(k + 1) * args.nq, :]
                multi_shot = self.multi_shot_learner(mapped_unseen_ent_embs[k * args.few:(k + 1) * args.few, :], que,
                                                     sup_time_cur, self.config['support_learner'])
                if self.config['support_learner'] == 'transformer':
                    multi_shot = multi_shot.squeeze(0)
                multi_shot_support_for_query.append(multi_shot)
                current_uent = uent
                k += 1
        return torch.cat(multi_shot_support_for_query, dim=0)

    def entity_learner_eval(self, support, query, query_count, args):
        current_uent = None
        k = 0
        multi_shot_support_for_query = []
        if args.cuda:
            unseen_ent = support[:, 0].cuda()
            sup_rel = support[:, 1].cuda()
            sup_obj = support[:, 2].cuda()
            sup_time = support[:, 3].cuda()
        else:
            unseen_ent = support[:, 0]
            # print(unseen_ent)
            sup_rel = support[:, 1]
            sup_obj = support[:, 2]
            sup_time = support[:, 3]
        mapped_unseen_ent_embs = self.entity_map(unseen_ent, sup_rel, sup_obj, sup_time, model=self.config['entity_learner'])

        for i, uent in enumerate(unseen_ent):
            if uent != current_uent:
                sup_time_cur = support[:, 3][k * args.few:(k + 1) * args.few]
                if k == 0:
                    if args.cuda:
                        que = query[:query_count[k], :].cuda()
                    else:
                        que = query[:query_count[k], :]
                else:
                    if args.cuda:
                        que = query[query_count[k - 1]:query_count[k], :].cuda()
                    else:
                        que = query[query_count[k - 1]:query_count[k], :]
                multi_shot = self.multi_shot_learner(mapped_unseen_ent_embs[k * args.few:(k + 1) * args.few, :], que,
                                                     sup_time_cur, self.config['support_learner'])
                if self.config['support_learner'] == 'transformer':
                    multi_shot = multi_shot.squeeze(0)
                multi_shot_support_for_query.append(multi_shot)
                current_uent = uent
                k += 1
        return torch.cat(multi_shot_support_for_query, dim=0)

    def entity_map(self, unseen_ent, sup_rel, sup_obj, sup_time, model=None):
        sup_rels = []
        for i in range(sup_rel.shape[0]):
            r = sup_rel[i].item()
            if r < self.num_rel:
                sup_rels.append(r)
            else:
                sup_rels.append(r + 1)
        if self.config['cuda']:
            sup_rels = torch.tensor(sup_rels).cuda()
        else:
            sup_rels = torch.tensor(sup_rels)
        sup_rel_emb = self.agent.rel_embs(sup_rels)
        sup_obj_emb = self.agent.ent_embs.ent_embs(sup_obj)
        if model == 'complex':
            pass
        elif model == 'nn':
            return self.entity_learning_layer(torch.cat((sup_obj_emb, sup_rel_emb), dim=1))
        elif model == 'dnn':
            return self.entity_learning_layer(torch.cat((self.proj1(sup_obj_emb + sup_rel_emb), self.proj2(sup_obj_emb * sup_rel_emb)), dim=1))

    def multi_shot_learner(self, multi_shot_raw, que, sup_time_cur, method='mean'):
        if method == 'mean':
            return torch.mean(multi_shot_raw, dim=0).repeat(que.shape[0], 1)
        elif method == 'transformer':
            if self.config['cuda']:
                que_time_expand = que[:, 3].unsqueeze(0).cuda()
                sup_time_cur_exapnd = sup_time_cur.unsqueeze(0).cuda()
            else:
                que_time_expand = que[:, 3].unsqueeze(0)
                sup_time_cur_exapnd = sup_time_cur.unsqueeze(0)
            multi_shot_raw_expand = multi_shot_raw.unsqueeze(0)
            return self.transformer(multi_shot_raw_expand, que_time_expand, sup_time_cur_exapnd)

    def forward(self, query_entities, query_entities_embeds, query_timestamps, query_relations, query_answers_embeds,
                issupport=False):
        """
        Args:
            query_entities: [batch_size]
            query_timestamps: [batch_size]
            query_relations: [batch_size]
        Return:
            all_loss: list
            all_logits: list
            all_actions_idx: list
            current_entities: torch.tensor, [batch_size]
            current_timestamps: torch.tensor, [batch_size]
        """
        query_relations_embeds = self.agent.rel_embs(query_relations)

        current_entites = query_entities
        current_timestamps = query_timestamps
        prev_relations = torch.ones_like(query_relations) * self.num_rel  # NO_OP

        all_loss = []
        all_logits = []
        all_actions_idx = []
        all_loss_reg = []
        all_reward = []


        self.agent.policy_step.set_hiddenx(query_relations.shape[0])
        for t in range(self.path_length):
            if t == 0:
                first_step = True
            else:
                first_step = False

            if t == self.path_length - 1:
                last_step = True
            else:
                last_step = False

            action_space = self.env.next_actions(
                current_entites,
                current_timestamps,
                query_timestamps,
                self.max_action_num,
                first_step
            )

            loss, logits, action_id, loss_sec_regularize, reward = self.agent(
                prev_relations,
                current_entites,
                current_timestamps,
                query_relations_embeds,
                query_entities_embeds,
                query_timestamps,
                action_space,
                first_step,
                last_step,
                query_relations,
                answer_embeds=query_answers_embeds,
                testing=False
            )

            chosen_relation = torch.gather(action_space[:, :, 0], dim=1, index=action_id).reshape(action_space.shape[0])
            chosen_entity = torch.gather(action_space[:, :, 1], dim=1, index=action_id).reshape(action_space.shape[0])
            chosen_entity_timestamps = torch.gather(action_space[:, :, 2], dim=1, index=action_id).reshape(
                action_space.shape[0])

            all_loss.append(loss)
            all_logits.append(logits)
            all_actions_idx.append(action_id)
            all_loss_reg.append(loss_sec_regularize)
            all_reward.append(reward)

            current_entites = chosen_entity
            current_timestamps = chosen_entity_timestamps
            prev_relations = chosen_relation

        return all_loss, all_logits, all_actions_idx, current_entites, current_timestamps, all_loss_reg, all_reward

    def beam_search(self, query_entities, query_timestamps, query_relations, query_entity_embeds,
                    query_answer_embeds=None):
        """
        Args:
            query_entities: [batch_size]
            query_timestamps: [batch_size]
            query_relations: [batch_size]
        Return:
            current_entites: [batch_size, test_rollouts_num]
            beam_prob: [batch_size, test_rollouts_num]
        """
        batch_size = query_entities.shape[0]
        query_entities_embeds = query_entity_embeds
        query_relations_embeds = self.agent.rel_embs(query_relations)

        self.agent.policy_step.set_hiddenx(batch_size)

        # In the first step, if rollouts_num is greater than the maximum number of actions, select all actions
        current_entites = query_entities
        current_timestamps = query_timestamps
        prev_relations = torch.ones_like(query_relations) * self.num_rel  # NO_OP
        action_space = self.env.next_actions(current_entites, current_timestamps,
                                             query_timestamps, self.max_action_num, True)
        loss, logits, action_id, _, reward = self.agent(
            prev_relations,
            current_entites,
            current_timestamps,
            query_relations_embeds,
            query_entities_embeds,
            query_timestamps,
            action_space,
            first_step=True,
            last_step=False,
            query_rel_id=query_relations,
            answer_embeds=query_answer_embeds,
            testing=True
        )  # logits.shape: [batch_size, max_action_num]

        action_space_size = action_space.shape[1]
        if self.config['beam_size'] > action_space_size:
            beam_size = action_space_size
        else:
            beam_size = self.config['beam_size']
        beam_log_prob, top_k_action_id = torch.topk(logits, beam_size,
                                                    dim=1)  # beam_log_prob.shape [batch_size, beam_size]
        beam_log_prob = beam_log_prob.reshape(-1)  # [batch_size * beam_size]

        current_entites = torch.gather(action_space[:, :, 1], dim=1, index=top_k_action_id).reshape(
            -1)  # [batch_size * beam_size]
        current_timestamps = torch.gather(action_space[:, :, 2], dim=1, index=top_k_action_id).reshape(
            -1)  # [batch_size * beam_size]
        prev_relations = torch.gather(action_space[:, :, 0], dim=1, index=top_k_action_id).reshape(
            -1)  # [batch_size * beam_size]

        if self.config['history_encoder'] == 'gru':
            self.agent.policy_step.hx = self.agent.policy_step.hx.repeat(1, 1, beam_size).reshape(
                [batch_size * beam_size, -1])  # [batch_size * beam_size, state_dim]
        else:
            self.agent.policy_step.hx = self.agent.policy_step.hx.repeat(1, 1, beam_size).reshape(
                [batch_size * beam_size, -1])  # [batch_size * beam_size, state_dim]
            self.agent.policy_step.cx = self.agent.policy_step.cx.repeat(1, 1, beam_size).reshape(
                [batch_size * beam_size, -1])  # [batch_size * beam_size, state_dim]

        beam_tmp = beam_log_prob.repeat([action_space_size, 1]).transpose(1, 0)  # [batch_size * beam_size, max_action_num]
        for t in range(1, self.path_length):
            if t == self.path_length - 1:
                last_step = True
                first_step = False
            else:
                last_step = False
                first_step = False
            query_timestamps_roll = query_timestamps.repeat(beam_size, 1).permute(1, 0).reshape(-1)
            query_entities_embeds_roll = query_entities_embeds.repeat(1, 1, beam_size)
            query_entities_embeds_roll = query_entities_embeds_roll.reshape(
                [batch_size * beam_size, -1])  # [batch_size * beam_size, ent_dim]
            query_relations_embeds_roll = query_relations_embeds.repeat(1, 1, beam_size)
            query_relations_embeds_roll = query_relations_embeds_roll.reshape(
                [batch_size * beam_size, -1])  # [batch_size * beam_size, rel_dim]

            action_space = self.env.next_actions(current_entites, current_timestamps,
                                                 query_timestamps_roll, self.max_action_num)

            loss, logits, action_id, loss_sec_regularize, reward = self.agent(
                prev_relations,
                current_entites,
                current_timestamps,
                query_relations_embeds_roll,
                query_entities_embeds_roll,
                query_timestamps_roll,
                action_space,
                first_step,
                last_step,
                query_rel_id=query_relations,
                answer_embeds=query_answer_embeds,
                testing=True
            )  # logits.shape [bs * rollouts_num, max_action_num]

            if self.config['history_encoder'] == 'gru':
                hx_tmp = self.agent.policy_step.hx.reshape(batch_size, beam_size, -1)
            else:
                hx_tmp = self.agent.policy_step.hx.reshape(batch_size, beam_size, -1)
                cx_tmp = self.agent.policy_step.cx.reshape(batch_size, beam_size, -1)

            beam_tmp = beam_log_prob.repeat([action_space_size, 1]).transpose(1, 0)  # [batch_size * beam_size, max_action_num]
            beam_tmp += logits
            beam_tmp = beam_tmp.reshape(batch_size, -1)  # [batch_size, beam_size * max_actions_num]

            if action_space_size * beam_size >= self.config['beam_size']:
                beam_size = self.config['beam_size']
            else:
                beam_size = action_space_size * beam_size

            top_k_log_prob, top_k_action_id = torch.topk(beam_tmp, beam_size, dim=1)  # [batch_size, beam_size]
            offset = top_k_action_id // action_space_size  # [batch_size, beam_size]
            offset = offset.unsqueeze(-1).repeat(1, 1, self.config['state_dim'])  # [batch_size, beam_size]
            if self.config['history_encoder'] == 'gru':
                self.agent.policy_step.hx = torch.gather(hx_tmp, dim=1, index=offset)
                self.agent.policy_step.hx = self.agent.policy_step.hx.reshape([batch_size * beam_size, -1])
            else:
                self.agent.policy_step.hx = torch.gather(hx_tmp, dim=1, index=offset)
                self.agent.policy_step.hx = self.agent.policy_step.hx.reshape([batch_size * beam_size, -1])
                self.agent.policy_step.cx = torch.gather(cx_tmp, dim=1, index=offset)
                self.agent.policy_step.cx = self.agent.policy_step.cx.reshape([batch_size * beam_size, -1])

            current_entites = torch.gather(action_space[:, :, 1].reshape(batch_size, -1), dim=1,
                                           index=top_k_action_id).reshape(-1)
            current_timestamps = torch.gather(action_space[:, :, 2].reshape(batch_size, -1), dim=1,
                                              index=top_k_action_id).reshape(-1)
            prev_relations = torch.gather(action_space[:, :, 0].reshape(batch_size, -1), dim=1,
                                          index=top_k_action_id).reshape(-1)

            beam_log_prob = top_k_log_prob.reshape(-1)  # [batch_size * beam_size]


        return action_space[:, :, 1].reshape(batch_size, -1), beam_tmp

    def switch_env(self, env_new):
        self.env = env_new.cuda()
