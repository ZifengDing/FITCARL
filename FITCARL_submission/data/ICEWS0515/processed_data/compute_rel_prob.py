import numpy as np
import torch
import pickle

ent2sec = np.load('ent2sec_matrix.npy')
print(ent2sec.shape)
rel_prob = {}
rel_count = {}
relations = set()

with open('background_graph_for_rl.txt', 'r') as f:
    for line in f:
        line_split = line.split()
        s, r, o, t = line_split[0], line_split[1], line_split[2], line_split[3]
        s_sec = torch.tensor(ent2sec[int(s)])
        o_sec = torch.tensor(ent2sec[int(o)])
        relations.add(int(r))
        if int(r) not in rel_prob.keys():
            rel_prob.update({int(r):[torch.zeros(1, ent2sec.shape[1]), torch.zeros(1, ent2sec.shape[1])]})
            rel_count.update({int(r): 1})
            rel_inv = int(r) + 251 + 1
            rel_prob.update({rel_inv: [torch.zeros(1, ent2sec.shape[1]), torch.zeros(1, ent2sec.shape[1])]})
            rel_count.update({rel_inv: 1})
            continue

        rel_inv = int(r) + 251 + 1
        rel_prob[int(r)][0] += s_sec
        rel_prob[int(r)][1] += o_sec
        rel_count[int(r)] += 1
        rel_prob[rel_inv][1] += s_sec
        rel_prob[rel_inv][0] += o_sec
        rel_count[rel_inv] += 1

rel_prob.update({251:[torch.zeros(1, ent2sec.shape[1]), torch.zeros(1, ent2sec.shape[1])]})
rel_count.update({251: 1})
softmax = torch.nn.Softmax(dim=1)
rel2secprob = []
for rel in range(0, 251):
    if rel not in relations:
        print(rel)

for rel in sorted(rel_prob.keys()):
    dist = torch.tensor(rel_prob[rel][1])
    rel2secprob.append(dist/torch.tensor(rel_count[rel]))

rel2secprob = torch.cat(rel2secprob)

with open('rel2secprob.pickle', 'wb') as handle:
    pickle.dump(rel2secprob, handle)
            
