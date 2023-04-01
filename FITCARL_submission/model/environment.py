import networkx as nx
from collections import defaultdict
import numpy as np
import torch

class Env(object):
    def __init__(self, examples, config, state_action_space=None):
        """Temporal Knowledge Graph Environment.
        examples: quadruples (subject, relation, object, timestamps);
        config: config dict;
        state_action_space: Pre-processed action space;
        """
        self.config = config
        self.num_rel = config['num_rel']
        self.graph, self.label2nodes = self.build_graph(examples)
        # [0, num_rel) -> normal relations; num_rel -> stay in place，(num_rel, num_rel * 2] reversed relations.
        self.NO_OP = self.num_rel  # Stay in place; No Operation
        self.ePAD = config['num_ent']  # Padding entity
        self.rPAD = config['num_rel'] * 2 + 1  # Padding relation.
        self.tPAD = 0  # Padding time
        self.state_action_space = state_action_space  # Pre-processed action space
        if state_action_space:
            self.state_action_space_key = self.state_action_space.keys()

        self.back_ent = None
        self.affected_back_ent = None

        if config['adaptive_sample']:
            self.time_weight = None
            self.time_encoder = None

    def adjust_graph(self, old_support, new_support):
        """
        For adjusting graph when we sample new support set in meta training.
        :param old_support:
        :param new_support:
        :return:
        """
        affected_back_ent_remove = self.remove_oldfew(old_support)
        affected_back_ent_add = self.add_newfew(new_support)

        self.affected_back_ent = affected_back_ent_add.union(affected_back_ent_remove)

    def add_newfew(self, quads):
        affected_back_ent = set()
        for q in quads:
            src_node, dst_node, rel = (q[0].item(), q[3].item()), (q[2].item(), q[3].item()), q[1].item()
            self.graph.add_node(src_node, label=q[0].item())
            self.graph.add_node(dst_node, label=q[2].item())
            if rel >= self.num_rel:
                self.graph.add_edge(src_node, dst_node, relation=rel + 1)
                self.graph.add_edge(dst_node, src_node, relation=rel - self.num_rel)
            else:
                self.graph.add_edge(src_node, dst_node, relation=rel)
                self.graph.add_edge(dst_node, src_node, relation=rel + self.num_rel + 1)
            self.label2nodes[q[0].item()].add((q[0].item(), q[3].item()))
            self.label2nodes[q[2].item()].add((q[2].item(), q[3].item()))
            affected_back_ent.add(q[2].item())
        return affected_back_ent

    def remove_oldfew(self, quads):
        affected_back_ent = set()
        for q in quads:
            src_node, dst_node = (q[0].item(), q[3].item()), (q[2].item(), q[3].item())
            if src_node in self.graph.nodes:
                self.graph.remove_node(src_node)
            self.label2nodes[q[0].item()] = set()
            affected_back_ent.add(q[2].item())
        return affected_back_ent

    def remove_oldfew_edges(self, quads):
        exist_connection = defaultdict(int)
        remove_list = []
        for q in quads:
            src, dst = (q[0],q[3]), (q[2],q[3])
            if (src, dst) in exist_connection.keys():
                exist_connection[(src, dst)] += 1
                remove_list.append([src, dst, exist_connection[(src, dst)]])
            else:
                exist_connection[(src, dst)] = 0
                remove_list.append([src, dst, 0])
                # count += 1
        return remove_list

    def build_graph(self, examples):
        """The graph node is represented as (entity, time), and the edges are directed and labeled relation.
        return:
            graph: nx.MultiDiGraph;
            label2nodes: a dict [keys -> entities, value-> nodes in the graph (entity, time)]
        """
        graph = nx.MultiDiGraph()
        label2nodes = defaultdict(set)
        examples.sort(key=lambda x: x[3], reverse=True)  # Reverse chronological order
        for example in examples:
            src = example[0]
            rel = example[1]
            dst = example[2]
            time = example[3]

            # Add the nodes and edges of the current quadruple
            src_node = (src, time)
            dst_node = (dst, time)
            if src_node not in label2nodes[src]:
                graph.add_node(src_node, label=src)
            if dst_node not in label2nodes[dst]:
                graph.add_node(dst_node, label=dst)

            graph.add_edge(src_node, dst_node, relation=rel)
            graph.add_edge(dst_node, src_node, relation=rel+self.num_rel+1)

            label2nodes[src].add(src_node)
            label2nodes[dst].add(dst_node)
        return graph, label2nodes

    def remove_old_support(self, old_support):
        for q in old_support:
            if q[1] < self.num_rel: # not inverse; (s, r, o, t), unseen entity is s
                # del (r, o, t)
                for i, a in enumerate(self.state_action_space[(q[0], q[3], True)]):
                    if list(a) == [q[1], q[2], q[3]]:
                        # print('< r, o, t:', a, [q[1], q[2], q[3]])
                        self.state_action_space[(q[0], q[3], True)] = np.delete(self.state_action_space[(q[0], q[3], True)], i, 0)
                # del (r+num_rel, s, t)
                for i, a in enumerate(self.state_action_space[(q[2], q[3], True)]):
                    # print(self.state_action_space[(q[2], q[3], True)])
                    if list(a) == [q[1] + self.num_rel + 1, q[0], q[3]]:
                        # print('< r+num_rel, s, t:', a, [q[1] , q[0], q[3]])
                        self.state_action_space[(q[2], q[3], True)] = np.delete(self.state_action_space[(q[2], q[3], True)], i, 0)
            else: # inverse; (o, r+num_rel, s, t), unseen entity is o
                # del (r+num_rel, s, t)
                for i, a in enumerate(self.state_action_space[(q[0], q[3], True)]):
                    if list(a) == [q[1] + 1, q[2], q[3]]:
                        # print('> r+num_rel, s, t:', a, [q[1], q[2], q[3]])
                        self.state_action_space[(q[0], q[3], True)] = np.delete(self.state_action_space[(q[0], q[3], True)], i, 0)
                # del (r, o, t)
                for i, a in enumerate(self.state_action_space[(q[2], q[3], True)]):
                    if list(a) == [q[1] - self.num_rel, q[0], q[3]]:
                        # print('> r,o,t:', a, [q[1], q[0], q[3]])
                        self.state_action_space[(q[2], q[3], True)] = np.delete(self.state_action_space[(q[2], q[3], True)], i, 0)

    def add_new_support(self, new_support):
        for q in new_support:
            if q[1] < self.num_rel: # not inverse; (s, r, o, t), unseen entity is s
                # add (r, o, t)
                if (q[0], q[3], True) not in self.state_action_space.keys():
                    self.state_action_space[(q[0], q[3], True)] = np.array(list([[q[1], q[2], q[3]]]), dtype=np.dtype('int32'))
                    # print(self.state_action_space[(q[0], q[3], True)])
                else:
                    self.state_action_space[(q[0], q[3], True)] = np.concatenate((self.state_action_space[(q[0], q[3], True)], [[q[1], q[2], q[3]]]), axis=0)

                # add (r+num_rel, s, t)
                if (q[2], q[3], True) not in self.state_action_space.keys():
                    self.state_action_space[(q[2], q[3], True)] = np.array(list([[q[1] + self.num_rel + 1, q[0], q[3]]]), dtype=np.dtype('int32'))
                    # print(self.state_action_space[(q[2], q[3], True)])
                else:
                    self.state_action_space[(q[2], q[3], True)] = np.concatenate((self.state_action_space[(q[2], q[3], True)], [[q[1] + self.num_rel + 1, q[0], q[3]]]), axis=0)
                    # print(self.state_action_space[(q[2], q[3], True)])

            else: # inverse; (o, r+num_rel, s, t), unseen entity is o
                # add (r+num_rel, s, t)
                if (q[0], q[3], True) not in self.state_action_space.keys():
                    self.state_action_space[(q[0], q[3], True)] = np.array(list([[q[1] + 1, q[2], q[3]]]), dtype=np.dtype('int32'))
                else:
                    self.state_action_space[(q[0], q[3], True)] = np.concatenate((self.state_action_space[(q[0], q[3], True)], [[q[1] + 1, q[2], q[3]]]), axis=0)

                # add (r, o, t)
                if (q[2], q[3], True) not in self.state_action_space.keys():
                    self.state_action_space[(q[2], q[3], True)] = np.array(list([[q[1] - self.num_rel, q[0], q[3]]]), dtype=np.dtype('int32'))
                else:
                    self.state_action_space[(q[2], q[3], True)] = np.concatenate((self.state_action_space[(q[2], q[3], True)], [[q[1] - self.num_rel, q[0], q[3]]]), axis=0)

    def adjust_state_actions_space(self, old_support, new_support, actions_space=None):
        self.add_new_support(new_support)
        self.remove_old_support(old_support)
        self.state_action_space_key = self.state_action_space.keys()

    def time_aware_action_sample(self, nodes, time):
        nodes = list(nodes)
        nodes.sort(key=lambda x: abs(x[1] - time), reverse=False)
        return nodes

    def set_time_encoder(self, time_encoder, time_weight):
        self.time_encoder = time_encoder
        self.time_weight = time_weight

    def random_sample(self, nodes):
        nodes_ = list(nodes)
        nodes_ = torch.tensor(nodes_)
        indices = torch.randperm(len(nodes_))
        nodes_sorted = nodes_[indices]
        return nodes_sorted

    def random_sample_(self, state_action_space):
        state_action_space_ = torch.tensor(state_action_space)
        indices = torch.randperm(state_action_space.shape[0])
        nodes_sorted = state_action_space_[indices]
        return nodes_sorted

    def adaptive_sample(self, nodes, time):
        nodes_ = list(nodes)
        node_time = torch.tensor([x[1] for x in nodes_]).unsqueeze(-1).cuda() if self.config['cuda'] else torch.tensor([x[1] for x in nodes_]).unsqueeze(-1)
        time_ = torch.tensor([time] * len(nodes_)).unsqueeze(-1).cuda() if self.config['cuda'] else torch.tensor([time] * len(nodes_)).unsqueeze(-1)
        node_time_emb = self.time_encoder.forward_transformer(time_ - node_time)
        node_time_score = torch.matmul(node_time_emb, self.time_weight).squeeze(1)
        nodes_ = torch.tensor(nodes_)
        node_prob = torch.softmax(node_time_score, dim=0)
        nodes_sorted = nodes_[torch.argsort(node_prob, descending=True)]
        return nodes_sorted

    def adaptive_sample_(self, state_action_space, time):
        node_time = torch.tensor(state_action_space[:, 2]).unsqueeze(-1).cuda() if self.config['cuda'] else torch.tensor(state_action_space[:, 2]).unsqueeze(-1)
        time_ = torch.tensor([time] * state_action_space.shape[0]).unsqueeze(-1).cuda() if self.config['cuda'] else torch.tensor([time] * state_action_space.shape[0]).unsqueeze(-1)
        node_time_emb = self.time_encoder.forward_transformer(time_ - node_time)
        node_time_score = torch.matmul(node_time_emb, self.time_weight).squeeze(1)
        state_action_space_ = torch.tensor(state_action_space)
        node_prob = torch.softmax(node_time_score, dim=0)
        nodes_sorted = state_action_space_[torch.argsort(node_prob, descending=True)]
        return nodes_sorted

    def time_aware_action_sample_(self, state_action_space, time):
        node_time = state_action_space[:, 2]
        time_ = np.array([time] * state_action_space.shape[0])
        time_abs = np.abs(time_ - node_time)
        nodes_sorted = state_action_space[np.argsort(time_abs, axis=0)]
        return nodes_sorted

    def expand_state_action_space(self, state_action_space, support):
        if support[1] < self.num_rel:
            state_action_space_ = np.concatenate((state_action_space, [[support[1].item() + self.num_rel + 1, support[0].item(), support[3].item()]]), axis=0)
        else:
            state_action_space_ = np.concatenate(
                (state_action_space, [[support[1].item() - self.num_rel, support[0].item(), support[3].item()]]), axis=0)
        return state_action_space_

    def expand_state_action_space_all(self, supports):
        for q in supports:
            if (q[2].item(), q[3].item(), True) in self.state_action_space_key:
                if q[2].item() in self.back_ent:
                    state_action_space = self.state_action_space[(q[2].item(), q[3].item(), True)]
                    self.state_action_space[(q[2].item(), q[3].item(), True)] = self.expand_state_action_space(state_action_space, q)

    def remove_state_action_space_all(self, old_supports):
        for q in old_supports:
            if (q[2].item(), q[3].item(), True) in self.state_action_space_key:
                if q[2].item() in self.back_ent:
                    state_action_space = self.state_action_space[(q[2].item(), q[3].item(), True)]
                    self.state_action_space[(q[2].item(), q[3].item(), True)] = state_action_space[:-self.config['few'], :]

    def get_state_actions_space_complete(self, entity, time, current_=True, max_action_num=None):
        """Get the action space of the current state.
        Args:
            entity: The entity of the current state;
            time: Maximum timestamp for candidate actions;
            current_: Can the current time of the event be used;
            max_action_num: Maximum number of events stored;
        Return:
            numpy array，shape: [number of events，3], (relation, dst, time)
        """
        if self.state_action_space:
            if (entity, time, current_) in self.state_action_space_key:
                if self.config['adaptive_sample']:
                    return self.adaptive_sample_(self.state_action_space[(entity, time, current_)], time)
                elif self.config['random_sample']:
                    return self.random_sample_(self.state_action_space[(entity, time, current_)])
                else:
                    return self.time_aware_action_sample_(self.state_action_space[(entity, time, current_)], time)
        nodes = self.label2nodes[entity].copy()
        if self.config['adaptive_sample']:
            nodes = self.adaptive_sample(nodes, time)
        elif self.config['random_sample']:
            nodes = self.random_sample(nodes)
        else:
            nodes = self.time_aware_action_sample(nodes, time)
        actions_space = []
        i = 0
        for node in nodes:
            if self.config['adaptive_sample'] or self.config['random_sample']:
                node = (node[0].item(), node[1].item())
            for src, dst, rel in self.graph.out_edges(node, data=True):
                actions_space.append((rel['relation'], dst[0], dst[1]))
                i += 1
                if max_action_num and i >= max_action_num:
                    break
            if max_action_num and i >= max_action_num:
                break
        return np.array(list(actions_space), dtype=np.dtype('int32'))

    def next_actions(self, entites, times, query_times, max_action_num=200, first_step=False):
        """Get the current action space. There must be an action that stays at the current position in the action space.
        Args:
            entites: torch.tensor, shape: [batch_size], the entity where the agent is currently located;
            times: torch.tensor, shape: [batch_size], the timestamp of the current entity;
            query_times: torch.tensor, shape: [batch_size], the timestamp of query;
            max_action_num: The size of the action space;
            first_step: Is it the first step for the agent.
        Return: torch.tensor, shape: [batch_size, max_action_num, 3], (relation, entity, time)
        """
        if self.config['cuda']:
            entites = entites.cpu()
            times = times.cpu()
            query_times = times.cpu()

        entites = entites.numpy()
        times = times.numpy()
        query_times = query_times.numpy()

        actions = self.get_padd_actions(entites, times, query_times, max_action_num, first_step)

        if self.config['cuda']:
            actions = torch.tensor(actions, dtype=torch.long, device='cuda')
        else:
            actions = torch.tensor(actions, dtype=torch.long)
        return actions

    def get_padd_actions(self, entites, times, query_times, max_action_num=200, first_step=False):
        """Construct the model input array.
        If the optional actions are greater than the maximum number of actions, then sample,
        otherwise all are selected, and the insufficient part is pad.
        """
        actions = np.ones((entites.shape[0], max_action_num, 3), dtype=np.dtype('int32'))
        actions[:, :, 0] *= self.rPAD
        actions[:, :, 1] *= self.ePAD
        actions[:, :, 2] *= self.tPAD
        for i in range(entites.shape[0]):
            # NO OPERATION
            actions[i, 0, 0] = self.NO_OP
            actions[i, 0, 1] = entites[i]
            actions[i, 0, 2] = times[i]

            action_array = self.get_state_actions_space_complete(entites[i], times[i], True)

            if action_array.shape[0] == 0:
                continue

            # Whether to keep the action NO_OPERATION
            start_idx = 1
            if first_step:
                # The first step cannot stay in place
                start_idx = 0

            if action_array.shape[0] > (max_action_num - start_idx):
                # Sample. Take the first events.
                actions[i, start_idx:, ] = action_array[:max_action_num-start_idx]
            else:
                actions[i, start_idx:action_array.shape[0]+start_idx, ] = action_array
        return actions