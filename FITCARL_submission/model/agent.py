import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .modules import TransformerEncoder

class HistoryEncoder(nn.Module):
    def __init__(self, config, input_dim):
        super(HistoryEncoder, self).__init__()
        self.config = config
        self.lstm_cell = torch.nn.LSTMCell(input_size=input_dim,
                                           hidden_size=config['state_dim'])

    def set_hiddenx(self, batch_size):
        """Set hidden layer parameters. Initialize to 0"""
        if self.config['cuda']:
            self.hx = torch.zeros(batch_size, self.config['state_dim'], device='cuda')
            self.cx = torch.zeros(batch_size, self.config['state_dim'], device='cuda')
        else:
            self.hx = torch.zeros(batch_size, self.config['state_dim'])
            self.cx = torch.zeros(batch_size, self.config['state_dim'])

    def forward(self, prev_action, mask):
        """mask: True if NO_OP. ON_OP does not affect history coding results"""
        self.hx_, self.cx_ = self.lstm_cell(prev_action, (self.hx, self.cx))
        self.hx = torch.where(mask, self.hx, self.hx_)
        self.cx = torch.where(mask, self.cx, self.cx_)
        return self.hx

class HistoryEncoderGRU(nn.Module):
    def __init__(self, config):
        super(HistoryEncoderGRU, self).__init__()
        self.config = config
        self.gru_cell = torch.nn.GRUCell(input_size=config['action_dim'],
                                           hidden_size=config['state_dim'])

    def set_hiddenx(self, batch_size):
        """Set hidden layer parameters. Initialize to 0"""
        if self.config['cuda']:
            self.hx = torch.zeros(batch_size, self.config['state_dim'], device='cuda')
            # self.cx = torch.zeros(batch_size, self.config['state_dim'], device='cuda')
        else:
            self.hx = torch.zeros(batch_size, self.config['state_dim'])
            # self.cx = torch.zeros(batch_size, self.config['state_dim'])

    def forward(self, prev_action, mask):
        """mask: True if NO_OP. ON_OP does not affect history coding results"""
        self.hx_ = self.gru_cell(prev_action, self.hx)
        self.hx = torch.where(mask, self.hx, self.hx_)
        return self.hx

class PolicyMLP(nn.Module):
    def __init__(self, config):
        super(PolicyMLP, self).__init__()
        self.mlp_l1= nn.Linear(config['mlp_input_dim'], config['mlp_hidden_dim'], bias=True)
        self.mlp_l2 = nn.Linear(config['mlp_hidden_dim'], config['action_dim'], bias=True)

    def forward(self, state_query):
        hidden = torch.relu(self.mlp_l1(state_query))
        output = self.mlp_l2(hidden).unsqueeze(1)
        return output

class DynamicEmbedding(nn.Module):
    def __init__(self, n_ent, dim_ent, dim_t, config):
        super(DynamicEmbedding, self).__init__()
        self.ent_embs = nn.Embedding(n_ent, dim_ent - dim_t)
        if config['sector_emb']:
            self.ent_embs_copy = nn.Embedding(n_ent, dim_ent - dim_t)
            if config['emb_nograd']:
                self.ent_embs_copy.weight.requires_grad = False
        if config['emb_nograd']:
            self.ent_embs.weight.requires_grad = False
        self.w = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim_t))).float())
        self.b = torch.nn.Parameter(torch.zeros(dim_t).float())

    def forward(self, entities, dt):
        dt = dt.unsqueeze(-1)
        batch_size = dt.size(0)
        seq_len = dt.size(1)

        dt = dt.view(batch_size, seq_len, 1)
        t = torch.cos(self.w.view(1, 1, -1) * dt + self.b.view(1, 1, -1))
        t = t.squeeze(1)  # [batch_size, time_dim]

        e = self.ent_embs(entities)
        return torch.cat((e, t), -1)

    def forward_withraw(self, raw_feature, dt):
        dt = dt.unsqueeze(-1)
        batch_size = dt.size(0)
        seq_len = dt.size(1)

        dt = dt.view(batch_size, seq_len, 1)
        t = torch.cos(self.w.view(1, 1, -1) * dt + self.b.view(1, 1, -1))
        t = t.squeeze(1)  # [batch_size, time_dim]

        return torch.cat((raw_feature, t), -1)

    def forward_transformer(self, ts):
        B = ts.size(0)
        L = ts.size(1)
        map_ts = ts.unsqueeze(2) * self.w.view(1, 1, -1)
        map_ts += self.b.view(1, 1, -1)
        harmonic = torch.cos(map_ts)
        return harmonic

class StaticEmbedding(nn.Module):
    def __init__(self, n_ent, dim_ent, config):
        super(StaticEmbedding, self).__init__()
        self.ent_embs = nn.Embedding(n_ent, dim_ent)
        if config['emb_nograd']:
            self.ent_embs.weight.requires_grad = False

    def forward(self, entities, timestamps=None):
        return self.ent_embs(entities)

class Agent(nn.Module):
    def __init__(self, config):
        super(Agent, self).__init__()
        self.num_rel = config['num_rel'] * 2 + 2
        self.config = config

        # [0, num_rel) -> normal relations; num_rel -> stay in place，(num_rel, num_rel * 2] reversed relations.
        self.NO_OP = self.num_rel  # Stay in place; No Operation
        self.ePAD = config['num_ent']  # Padding entity
        self.rPAD = config['num_rel'] * 2 + 1 # Padding relation
        self.tPAD = 0  # Padding time

        if self.config['entities_embeds_method'] == 'dynamic':
            self.ent_embs = DynamicEmbedding(config['num_ent']+1, config['ent_dim'], config['time_dim'], config)
        else:
            self.ent_embs = StaticEmbedding(config['num_ent']+1, config['ent_dim'], config)

        self.rel_embs = nn.Embedding(config['num_ent'], config['rel_dim'])
        if config['emb_nograd']:
            self.rel_embs.weight.requires_grad = False

        if self.config['history_encoder'] == 'gru':
            self.policy_step = HistoryEncoderGRU(config)
        else:
            self.policy_step = HistoryEncoder(config, config['action_dim'])

        self.score_weight = torch.nn.Linear(config['ent_dim'], 1)
        self.score_mat = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (config['ent_dim'], config['ent_dim'])),
                                            dtype=torch.float, requires_grad=True))
        self.time_weight = nn.Parameter(torch.ones(config['time_dim']) * 0.001)
        self.proj_action = torch.nn.Linear(self.config['state_dim'], self.config['ent_dim'])
        self.proj_neighbor = torch.nn.Linear(self.config['ent_dim'] + self.config['rel_dim'], self.config['ent_dim'])
        self.proj_que = torch.nn.Linear(self.config['ent_dim'] + self.config['rel_dim'], self.config['ent_dim'])

        self.rel2secprob = None
        self.ent2sec = None
        self.sec_regularize = torch.nn.KLDivLoss(reduce=False, log_target=True)

        if self.config['conf'] and self.config['conf_mode'] == 'tucker':
            self.W_tk = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (config['ent_dim'], config['rel_dim'], config['ent_dim'])),
                                                dtype=torch.float, requires_grad=True))
            self.input_dropout = torch.nn.Dropout(0.1)
            self.hidden_dropout1 = torch.nn.Dropout(0.1)
            self.hidden_dropout2 = torch.nn.Dropout(0.1)

            self.bn0 = torch.nn.BatchNorm1d(config['ent_dim'])
            self.bn1 = torch.nn.BatchNorm1d(config['ent_dim'])

    def forward(self, prev_relation, current_entities, current_timestamps,
                query_relation, query_entity, query_timestamps, action_space, first_step, last_step, query_rel_id, answer_embeds=None, testing=False):
        """
        Args:
            prev_relation: [batch_size]
            current_entities: [batch_size]
            current_timestamps: [batch_size]
            query_relation: embeddings of query relation，[batch_size, rel_dim]
            query_entity: embeddings of query entity, [batch_size, ent_dim]
            query_timestamps: [batch_size]
            action_space: [batch_size, max_actions_num, 3] (relations, entities, timestamps)
        """
        # embeddings
        current_delta_time = query_timestamps - current_timestamps
        if not first_step:
            current_embds = self.ent_embs(current_entities, current_delta_time)  # [batch_size, ent_dim]
        else:
            current_embds = query_entity # first step use the entity learner output
        prev_relation_embds = self.rel_embs(prev_relation)  # [batch_size, rel_dim]

        # Pad Mask
        pad_mask = torch.ones_like(action_space[:, :, 0]) * self.rPAD  # [batch_size, action_number]
        pad_mask = torch.eq(action_space[:, :, 0], pad_mask)  # [batch_size, action_number]

        # History Encode
        NO_OP_mask = torch.eq(prev_relation, torch.ones_like(prev_relation) * self.NO_OP)  # [batch_size]
        NO_OP_mask = NO_OP_mask.repeat(self.config['state_dim'], 1).transpose(1, 0)  # [batch_size, state_dim]


        prev_action_embedding = torch.cat([prev_relation_embds, current_embds],
                                          dim=-1)  # [batch_size, rel_dim + ent_dim]
        lstm_output = self.policy_step(prev_action_embedding, NO_OP_mask)  # [batch_size, state_dim]

        # Neighbor/condidate_actions embeddings
        action_num = action_space.size(1)
        neighbors_delta_time = query_timestamps.unsqueeze(-1).repeat(1, action_num) - action_space[:, :, 2]
        neighbors_entities = self.ent_embs(action_space[:, :, 1], neighbors_delta_time)  # [batch_size, action_num, ent_dim]
        neighbors_relations = self.rel_embs(action_space[:, :, 0])  # [batch_size, action_num, rel_dim]

        neighbor = self.proj_neighbor(torch.cat([neighbors_relations, neighbors_entities], dim=-1)) # [batch_size, action_num, ent_dim]
        path = self.proj_action(lstm_output).unsqueeze(1) # [batch_size, 1, ent_dim]
        que = self.proj_que(torch.cat([query_relation, query_entity], dim=-1)).unsqueeze(1) # [batch_size, 1, ent_dim]

        time_diff_path = action_space[:, :, 2] - current_timestamps.unsqueeze(-1).repeat(1, action_num) # [batch_size, action_num]
        time_diff_que = action_space[:, :, 2] - query_timestamps.unsqueeze(-1).repeat(1, action_num) # [batch_size, action_num]
        time_path = self.ent_embs.forward_transformer(time_diff_path)
        time_que = self.ent_embs.forward_transformer(time_diff_que)

        path_score = torch.sum(neighbor * path, dim=2).unsqueeze(-1) # [batch_size, action_num, 1]
        que_score = torch.sum(neighbor * que, dim=2).unsqueeze(-1) # [batch_size, action_num, 1]
        time_path_score = torch.matmul(time_path, self.time_weight).unsqueeze(-1)
        time_que_score = torch.matmul(time_que, self.time_weight).unsqueeze(-1)

        action_att = torch.softmax(torch.cat([path_score + time_path_score, que_score + time_que_score], dim=-1), dim=-1) # [batch_size, action_num, 2]
        action_feature = action_att[:, :, 0].unsqueeze(-1) * path + action_att[:, :, 1].unsqueeze(-1) * que # [batch_size, action_num, ent_dim]

        scores = torch.sum((neighbor @ self.score_mat) * action_feature, dim=-1)

        # Padding mask
        scores = scores.masked_fill(pad_mask, -1e10)  # [batch_size ,action_number]

        action_prob = torch.softmax(scores, dim=1)
        action_prob_without_belief = action_prob
        ##### Confidence ##############
        if self.config['conf']:
            conf_prob = self.confidence(query_entity, query_relation, neighbors_entities, pad_mask, mode=self.config['conf_mode'])
            action_prob = torch.softmax(action_prob * conf_prob, dim=1)
        else:
            conf_prob = None
        ##### Concept Regularize ##############
        if self.config['sector']:
            if not testing:
                scores_sector = self.relsec_regularize(action_space, query_rel_id)
            else:
                scores_sector = None

        action_id = torch.multinomial(action_prob, 1)  # Randomly select an action. [batch_size, 1]

        if self.config['conf']:
            logits = torch.log(action_prob)  # [batch_size, action_number]
        else:
            logits = torch.nn.functional.log_softmax(scores, dim=1)
        one_hot = torch.zeros_like(logits).scatter(1, action_id, 1)
        loss = - torch.sum(torch.mul(logits, one_hot), dim=1)

        if self.config['sector']:
            if not testing:
                loss_sec_regularize = self.cal_sec_loss(logits, scores_sector)
            else:
                loss_sec_regularize = torch.zeros_like(loss)
        else:
            loss_sec_regularize = torch.zeros_like(loss)

        if not testing:
            if self.config['belief']:
                reward = self.get_reward_step(torch.gather(action_space[:, :, 1], dim=1, index=action_id).reshape(action_space.shape[0]), answer_embeds,
                                              torch.gather(conf_prob, dim=1, index=action_id).reshape(conf_prob.shape[0])).unsqueeze(1)
            else:
                reward = self.get_reward_step(
                    torch.gather(action_space[:, :, 1], dim=1, index=action_id).reshape(action_space.shape[0]),
                    answer_embeds,
                    None).unsqueeze(1)
        else:
            reward = None


        return loss, logits, action_id, loss_sec_regularize, reward
    
    def get_reward_step(self, chosen_entities, answer_embs, conf_chosen, current_ent_t_embs=None, src_que_embeds=None, rel_que_embeds=None, W_tk=None):
        chosen_ent_embs = self.ent_embs.ent_embs(chosen_entities)
        reward = torch.sigmoid(5-torch.norm(chosen_ent_embs - answer_embs, p=2, dim=1))
        return reward

    def cal_sec_loss(self, logits, scores_sector):
        action_prob = logits
        action_prob_sector = torch.nn.functional.log_softmax(scores_sector, dim=1)
        return torch.sum(self.sec_regularize(action_prob, action_prob_sector), dim=1)

    def relsec_regularize(self, action_space, query_rel_id):
        final_candidate = action_space[:, :, 1]
        final_candidate_sector = self.ent2sec[final_candidate]
        query_rel_id2sectorprob = self.rel2secprob[query_rel_id,:]
        query_rel_id2sectorprob = query_rel_id2sectorprob.unsqueeze(1)
        return torch.sum(query_rel_id2sectorprob * final_candidate_sector, dim=2)

    def confidence(self, query_entities, query_relations, neighbor_entities, pad_mask, mode='complex'):
        if mode == 'complex':
            rank = query_entities.shape[1] // 2
            lhs = query_entities[:, :rank], query_entities[:, rank:]
            rel = query_relations[:, :rank], query_relations[:, rank:]

            right = neighbor_entities
            right = right[:, :, :rank], right[:, :, rank:]
            s = (lhs[0] * rel[0] - lhs[1] * rel[1]).unsqueeze(1) @ right[0].transpose(1, 2) + \
                (lhs[0] * rel[1] + lhs[1] * rel[0]).unsqueeze(1) @ right[1].transpose(1, 2)
            s = s.squeeze(1).masked_fill(pad_mask, -1e10)
            s = torch.softmax(s, dim=1)
        elif mode == 'tucker':
            x = query_entities.view(-1, 1, self.config['ent_dim'])
            W_mat = torch.mm(query_relations, self.W_tk.view(self.config['rel_dim'], -1))
            W_mat = W_mat.view(-1, self.config['ent_dim'], self.config['ent_dim'])
            x = torch.bmm(x, W_mat)
            x = x.view(-1, self.config['ent_dim'])
            x = x.unsqueeze(1)
            s = x @ neighbor_entities.transpose(1, 2)
            s = s.squeeze(1).masked_fill(pad_mask, -1e10)
            s = torch.softmax(s, dim=1)
        return s

