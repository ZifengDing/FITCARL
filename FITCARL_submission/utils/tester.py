import torch
# import tqdm
import numpy as np
import random
import copy
from collections import defaultdict

class Tester(object):
    def __init__(self, model, args, train_entities, RelEntCooccurrence=None):
        self.model = model
        self.args = args
        self.train_entities = train_entities
        self.RelEntCooccurrence = RelEntCooccurrence
        self.curr_ent_idx = 0

    def get_rank(self, score, answer, entities_space, num_ent):
        if answer not in entities_space:
            rank = num_ent
            answer_prob = None
        else:
            answer_prob = score[entities_space.index(answer)]
            score.sort(reverse=True)
            rank = score.index(answer_prob) + 1
        return rank, score, answer_prob

    def test_fewshot(self, ent2quad, ntriple, skip_dict, skip_dict_unaware, num_ent, old_support):
        self.model.eval()
        logs = []
        time_aware_result = defaultdict(list)
        self.num_unseen_ent = len(list(ent2quad.keys()))
        self.all_unseen_entities = list(ent2quad.keys())
        self.tasks = ent2quad
        with torch.no_grad():
            [support, query], curr_unseen_ent, query_count = self.next_batch()
            support = torch.flatten(torch.tensor(support), start_dim=0, end_dim=1)
            query = torch.tensor(query)
            src_sup = support[:, 0]
            rel_sup = support[:, 1]
            dst_sup = support[:, 2]
            time_sup = support[:, 3]
            src_que = query[:, 0]
            rel_que = query[:, 1]
            dst_que = query[:, 2]
            time_que = query[:, 3]

            if self.args.cuda:
                src_sup = src_sup.cuda()
                rel_sup = rel_sup.cuda()
                dst_sup = dst_sup.cuda()
                time_sup = time_sup.cuda()
                src_que = src_que.cuda()
                rel_que = rel_que.cuda()
                dst_que = dst_que.cuda()
                time_que = time_que.cuda()

            if isinstance(old_support, np.ndarray):
                self.model.env.adjust_graph(old_support, support)
                self.model.env.remove_state_action_space_all(old_support)
                self.model.env.expand_state_action_space_all(support)
            else:
                affected_back_ent = self.model.env.add_newfew(support)
                self.model.env.affected_back_ent = affected_back_ent
                self.model.env.expand_state_action_space_all(support)

            if self.args.sector_emb:
                self.model.update_sector_emb()
                self.model.update_ent_emb_sector()

            unseen_ent_emb = self.model.entity_learner_eval(support, query, query_count, self.args)
            if self.args.sector_emb:
                unseen_ent_emb = self.model.update_ent_emb_sector_unseen(support, unseen_ent_emb)
            src_que_embeds = self.model.agent.ent_embs.forward_withraw(unseen_ent_emb, torch.zeros_like(time_que))

            group_num = 50
            cur_group_num = 0
            while cur_group_num * group_num < query.shape[0]:
                if (cur_group_num + 1) * group_num > query.shape[0]:
                    src_que_group, time_que_group, rel_que_group, dst_que_group = \
                        src_que[cur_group_num * group_num:], \
                        time_que[cur_group_num * group_num:], \
                        rel_que[cur_group_num * group_num:], \
                        dst_que[cur_group_num * group_num:]
                    src_que_embeds_group = src_que_embeds[cur_group_num * group_num:,:]
                    cur_group_num += 1
                else:
                    cur_group_num += 1
                    src_que_group, time_que_group, rel_que_group, dst_que_group = \
                        src_que[(cur_group_num-1) * group_num : cur_group_num * group_num], \
                        time_que[(cur_group_num-1) * group_num : cur_group_num * group_num], \
                        rel_que[(cur_group_num-1) * group_num : cur_group_num * group_num], \
                        dst_que[(cur_group_num-1) * group_num : cur_group_num * group_num]
                    src_que_embeds_group = src_que_embeds[(cur_group_num-1) * group_num : cur_group_num * group_num,:]

                current_entities, beam_prob = \
                    self.model.beam_search(src_que_group, time_que_group, rel_que_group, src_que_embeds_group)

                if self.args.cuda:
                    current_entities = current_entities.cpu()
                    beam_prob = beam_prob.cpu()

                current_entities = current_entities.numpy()
                beam_prob = beam_prob.numpy()

                MRR = 0

                for i in range(src_que_group.shape[0]):
                    hits1_id = []

                    candidate_answers = current_entities[i]
                    candidate_score = beam_prob[i]

                    idx = np.argsort(-candidate_score)
                    candidate_answers = candidate_answers[idx]
                    candidate_score = candidate_score[idx]

                    # remove duplicate entities
                    candidate_answers, idx = np.unique(candidate_answers, return_index=True)
                    candidate_answers = list(candidate_answers)
                    candidate_score = list(candidate_score[idx])

                    src = src_que_group[i].item()
                    rel = rel_que_group[i].item()
                    dst = dst_que_group[i].item()
                    time = time_que_group[i].item()

                    filter = skip_dict[(src, rel, time)]  # a set of ground truth entities
                    filter_unaware = skip_dict_unaware[(src, rel)]  # a set of ground truth entities, unaware

                    tmp_entities = candidate_answers.copy()
                    tmp_entities_unaware = candidate_answers.copy()
                    tmp_prob = candidate_score.copy()
                    tmp_prob_unaware = candidate_score.copy()

                    candidate_answers_unaware = candidate_answers.copy()
                    candidate_score_unaware = candidate_score.copy()

                    # time-aware filter
                    for j in range(len(tmp_entities)):
                        if tmp_entities[j] in filter and tmp_entities[j] != dst:
                            candidate_score[j] -= 1e10
                    ranking_raw, score_raw, answer_prob_raw = self.get_rank(candidate_score, dst, candidate_answers, num_ent)

                    # time-unaware filter
                    for j in range(len(tmp_entities_unaware)):
                        if tmp_entities_unaware[j] in filter_unaware and tmp_entities_unaware[j] != dst:
                            candidate_score_unaware[j] -= 1e10
                    ranking_un, score_un, answer_prob_un = self.get_rank(candidate_score_unaware, dst, candidate_answers_unaware, num_ent)

                    logs.append({
                        'MRR-aware': 1.0 / ranking_raw,
                        'MRR-unaware': 1.0 / ranking_un,
                        'HITS@1': 1.0 if ranking_un <= 1 else 0.0,
                        'HITS@3': 1.0 if ranking_un <= 3 else 0.0,
                        'HITS@10': 1.0 if ranking_un <= 10 else 0.0,
                    })
                    MRR = MRR + 1.0 / ranking_un

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        return metrics, support

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
        support_triples = [curr_tasks[i] for i in curr_tasks_idx[:self.args.few]]    # [F, 4]
        query_triples = [curr_tasks[i] for i in curr_tasks_idx[self.args.few:]]

        return support_triples, query_triples, curr_unseen_ent

    def next_batch(self):
        next_batch_all = [self.next_one() for _ in range(self.num_unseen_ent)]
        support, query, curr_unseen_ent = zip(*next_batch_all)
        support_fil = []
        query_fil = []
        query_count = []
        k = 0
        for i, q in enumerate(query):
            if len(q) == 0:
                continue
            else:
                support_fil.append(support[i])
                query_fil.append(query[i])
                query_count.append(len(query[i]))
                if k != 0:
                    query_count[-1] = query_count[-1] + query_count[-2]
                k += 1
        return [np.array(support_fil), np.concatenate([np.vstack(q) for q in query_fil], axis=0)], curr_unseen_ent, query_count
