import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.logger import *
from utils.trainer import Trainer
from utils.tester import Tester
from dataset.baseDataset import baseFSLDataset, QuadruplesDatasetFSL
from model.agent import Agent
from model.environment import Env
from model.episode import Episode
from model.policyGradient import PG
import os
import pickle
import random

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Temporal Knowledge Graph Forecasting Models',
        usage='main.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='whether to use GPU or not')
    parser.add_argument('--data_path', type=str, default='data/ICEWS14', help='path to data')
    parser.add_argument('--save_path', default='logs', type=str, help='log and model save path')
    parser.add_argument('--load_model_path', default='logs', type=str, help='trained model checkpoint path')
    parser.add_argument('--few', default=3, type=int, help='shot size of meta learning')
    parser.add_argument('--nq', default=10, type=int, help='size of query set in meta training')
    parser.add_argument('--support_learner', default='transformer', type=str, help='how to learn multi shot support')
    parser.add_argument('--sector', action='store_true', help='whether to use sector (concept) regularization')
    parser.add_argument('--sector_emb', action='store_true', help='whether to use sector (concept) embeddings')
    parser.add_argument('--history_encoder', default='gru', type=str, help='how to learn path embeddings')
    parser.add_argument('--score_module', default='att', type=str, help='which action scoring module')
    parser.add_argument('--entity_learner', default='nn', type=str, help='how to map entity from single support')
    parser.add_argument('--adaptive_sample', action='store_true', help='whether to use time-adaptive sample')
    parser.add_argument('--random_sample', action='store_true', help='whether to use random sample')

    # Train Params
    # parser.add_argument('--batch_size', default=512, type=int, help='training batch size.')
    parser.add_argument('--max_epochs', default=100000, type=int, help='max training epochs')
    parser.add_argument('--num_workers', default=8, type=int, help='workers number used for dataloader')
    parser.add_argument('--valid_epoch', default=10, type=int, help='validation frequency')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--save_epoch', default=100, type=int, help='model saving frequency')
    parser.add_argument('--clip_gradient', default=5.0, type=float, help='for gradient crop')
    parser.add_argument('--pretrain', action='store_true', help='whether to use pretrain')

    # Test Params
    parser.add_argument('--beam_size', default=100, type=int, help='the beam number of the beam search')

    # Agent Params
    parser.add_argument('--ent_dim', default=200, type=int, help='embedding dimension of the entities')
    parser.add_argument('--rel_dim', default=100, type=int, help='embedding dimension of the relations')
    parser.add_argument('--state_dim', default=100, type=int, help='dimension of the GRU hidden state')
    parser.add_argument('--hidden_dim', default=100, type=int, help='dimension of the hidden layer')
    parser.add_argument('--time_dim', default=100, type=int, help='embedding dimension of the timestamps')
    parser.add_argument('--entities_embeds_method', default='dynamic', type=str,
                        help='representation method of the entities, dynamic or static (dynamic is with time)')
    parser.add_argument('--emb_nograd', action='store_true', help='whether to stop gradient flow in entity and relation embedding')
    parser.add_argument('--conf', action='store_true',
                        help='whether to calculate confidence')
    parser.add_argument('--conf_mode', default='tucker', type=str,
                        help='confidence mode')

    # Environment Params
    parser.add_argument('--state_actions_path', default='state_actions_space_back.pkl', type=str,
                        help='the file stores preprocessed candidate action array')

    # Episode Params
    parser.add_argument('--path_length', default=3, type=int, help='the agent search path length')
    parser.add_argument('--max_action_num', default=50, type=int, help='the max candidate actions number')

    # Policy Gradient Params
    parser.add_argument('--Lambda', default=0.0, type=float, help='update rate of baseline')
    parser.add_argument('--Gamma', default=0.95, type=float, help='discount factor of Bellman equation')
    parser.add_argument('--Ita', default=0.01, type=float, help='regular proportionality constant')
    parser.add_argument('--Zita', default=0.9, type=float, help='attenuation factor of entropy regular term')

    return parser.parse_args(args)

def get_model_config(args, num_ent, num_rel):
    config = {
        'cuda': args.cuda,  # whether to use GPU or not.
        # 'batch_size': args.batch_size,  # training batch size.
        'num_ent': num_ent,  # number of entities
        'num_rel': num_rel,  # number of relations
        'ent_dim': args.ent_dim,  # embedding dimension of the entities
        'rel_dim': args.rel_dim,  # embedding dimension of the relations
        'time_dim': args.time_dim,  # embedding dimension of the timestamps
        'state_dim': args.state_dim,  # dimension of the GRU hidden state
        'action_dim': args.ent_dim + args.rel_dim,  # dimension of the actions
        'mlp_input_dim': args.ent_dim + args.rel_dim + args.state_dim,  # dimension of the input of the MLP
        'mlp_hidden_dim': args.hidden_dim,  # dimension of the MLP hidden layer
        'path_length': args.path_length,  # agent search path length
        'max_action_num': args.max_action_num,  # max candidate action number
        'lambda': args.Lambda,  # update rate of baseline
        'gamma': args.Gamma,  # discount factor of Bellman equation
        'ita': args.Ita,  # regular proportionality constant
        'zita': args.Zita,  # attenuation factor of entropy regular term
        'beam_size': args.beam_size,  # beam size for beam search
        'entities_embeds_method': args.entities_embeds_method,  # default: 'dynamic', otherwise static encoder will be used
        'emb_nograd': args.emb_nograd, # whether to  stop gradient flow in entity and relation embedding
        'support_learner': args.support_learner, # how to learn multi shot support
        'few': args.few, # shot size
        'sector': args.sector, # whether to use sector (concept) regularization
        'conf': args.conf, # whether to calculate confidence
        'conf_mode': args.conf_mode, # confidence mode
        'history_encoder': args.history_encoder, # how to learn path embeddings
        'score_module': args.score_module, # which action scoring module
        'sector_emb': args.sector_emb, # whether to use sector (concept) embeddings
        'adaptive_sample': args.adaptive_sample, # whether to use time-adaptive sample
        'random_sample': args.random_sample,  # whether to use random sample
        'entity_learner': args.entity_learner, # how to map entity from single support
    }
    return config

def main(args):
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

    #######################Set Logger#################################
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.cuda and torch.cuda.is_available():
        args.cuda = True
    else:
        args.cuda = False
    set_logger(args)

    #######################Create DataLoader#################################
    train_path = os.path.join(args.data_path, 'meta_train_all_for_rl.txt')
    test_path = os.path.join(args.data_path, 'meta_test_all_for_rl.txt')
    stat_path = os.path.join(args.data_path, 'stat.txt')
    valid_path = os.path.join(args.data_path, 'meta_valid_all_for_rl.txt')
    back_path = os.path.join(args.data_path, 'background_graph_for_rl.txt')
    backtrain_path = os.path.join(args.data_path, 'back_metatrain.txt')
    backvalid_path = os.path.join(args.data_path, 'back_metavalid.txt')
    backtest_path = os.path.join(args.data_path, 'back_metatest.txt')

    baseData_train = baseFSLDataset(train_path, stat_path, ismeta=True)
    baseData_valid = baseFSLDataset(valid_path, stat_path, ismeta=True)
    baseData_test = baseFSLDataset(test_path, stat_path, ismeta=True)

    baseData_backtrain = baseFSLDataset(backtrain_path, stat_path, ismeta=False)
    baseData_backvalid = baseFSLDataset(backvalid_path, stat_path, ismeta=False)
    baseData_backtest = baseFSLDataset(backtest_path, stat_path, ismeta=False)
    baseData_back = baseFSLDataset(back_path, stat_path, ismeta=False)

    train_entity = pickle.load(open(os.path.join(args.data_path, 'meta_train_ent.pkl'), 'rb'))
    valid_entity = pickle.load(open(os.path.join(args.data_path, 'meta_valid_ent.pkl'), 'rb'))
    test_entity = pickle.load(open(os.path.join(args.data_path, 'meta_test_ent.pkl'), 'rb'))

    train_ent2quad = pickle.load(open(os.path.join(args.data_path, 'meta_train_task_entity_to_quadruples_tohead.pickle'), 'rb'))
    valid_ent2quad = pickle.load(open(os.path.join(args.data_path, 'meta_valid_task_entity_to_quadruples_tohead.pickle'), 'rb'))
    test_ent2quad = pickle.load(open(os.path.join(args.data_path, 'meta_test_task_entity_to_quadruples_tohead.pickle'), 'rb'))

    trainDataset  = QuadruplesDatasetFSL(baseData_train.allQuadruples, baseData_train.num_r, train_entity)
    validDataset = QuadruplesDatasetFSL(baseData_valid.allQuadruples, baseData_valid.num_r, valid_entity)
    testDataset = QuadruplesDatasetFSL(baseData_test.allQuadruples, baseData_test.num_r, test_entity)

    ######################Creat the agent and the environment###########################
    config = get_model_config(args, baseData_train.num_e, baseData_train.num_r)
    logging.info(config)
    logging.info(args)

    # creat the agent
    agent = Agent(config)

    # creat the environment
    state_actions_path = os.path.join(args.data_path, args.state_actions_path)
    if not os.path.exists(state_actions_path):
        state_action_space = None
    else:
        state_action_space = pickle.load(open(os.path.join(args.data_path, args.state_actions_path), 'rb'))

    env = Env(baseData_back.allQuadruples, config, state_action_space) # base env, only contains background quadruples
    env.set_time_encoder(agent.ent_embs, agent.time_weight)
    back_ent = torch.tensor(
        list(pickle.load(open(os.path.join(args.data_path, 'background_entities.pkl'), 'rb'))))
    env.back_ent = back_ent


    # Create episode controller for meta training
    # env will be changed but agent remains the same
    episode = Episode(env, agent, config)
    # load pretrained embedding of background entities
    if args.pretrain:
        episode.load_pretrain(args.data_path + '/pretrain/ComplEx_entity_100.npy',
                              args.data_path + '/pretrain/ComplEx_relation_100.npy')
    if args.cuda:
        episode = episode.cuda()

    pg = PG(config)  # Policy Gradient
    optimizer = torch.optim.Adam(episode.parameters(), lr=args.lr, weight_decay=0.00001)

    # Load the model parameters
    if os.path.isfile(args.load_model_path):
        params = torch.load(args.load_model_path)
        episode.load_state_dict(params['model_state_dict'])
        optimizer.load_state_dict(params['optimizer_state_dict'])
        logging.info('Load trained model: {}'.format(args.load_model_path))

    # Load sector (concept) to enable type constraints for regularization
    if args.sector or args.sector_emb:
        sector_path = os.path.join(args.data_path, 'ent2sec_matrix.npy')
        ent2sec = np.load(sector_path)
        if args.cuda:
            ent2sec_ = torch.cat([torch.from_numpy(ent2sec), torch.zeros((1, ent2sec.shape[1]))], dim=0)
            episode.agent.ent2sec = ent2sec_.cuda()
            episode.agent.rel2secprob = pickle.load(
                open(os.path.join(args.data_path, 'rel2secprob.pickle'), 'rb')).cuda()
            if args.sector_emb:
                episode.ent2sec = ent2sec_.cuda()
                back_ent.cuda()
                episode.back_ent = back_ent
        else:
            ent2sec_ = torch.cat([torch.from_numpy(ent2sec), torch.zeros((1, ent2sec.shape[1]))], dim=0)
            episode.agent.ent2sec = ent2sec_
            episode.agent.rel2secprob = pickle.load(
                open(os.path.join(args.data_path, 'rel2secprob.pickle'), 'rb'))
            if args.sector_emb:
                episode.ent2sec = ent2sec_
                episode.back_ent = back_ent

    ######################Training and Testing###########################
    trainer = Trainer(episode, pg, optimizer, args)
    tester = Tester(episode, args, baseData_train.train_entities)

    best_mrr = 0
    best_epoch = 0

    logging.info('Start Training......')
    old_support = None
    for i in range(args.max_epochs):
        loss, reward, new_support = trainer.train_epoch(train_ent2quad, trainDataset.__len__(), old_support)
        old_support = new_support # update new sampled support
        logging.info('Epoch {}/{} Loss: {}, reward: {}'.format(i, args.max_epochs, loss, reward))

        if i % args.save_epoch == 0 and i != 0:
            trainer.save_model('checkpoint_{}.pth'.format(i))
            logging.info('Save Model in {}'.format(args.save_path))

        if i % args.valid_epoch == 0 and i != 0:
            logging.info('Start Val......')
            # episode.switch_env(env_valid)
            tester.model.env = trainer.model.env
            metrics, new_support = tester.test_fewshot(valid_ent2quad,
                                                       validDataset.__len__(),
                                                       baseData_backvalid.skip_dict,
                                                       baseData_backvalid.skip_dict_unaware,
                                                       config['num_ent'],
                                                       old_support)
            old_support = new_support
            for mode in metrics.keys():
                logging.info('Valid {} at epoch {}: {}'.format(mode, i, metrics[mode]))

            logging.info('Start Testing......')
            # episode.switch_env(env_test)
            tester.model.env = trainer.model.env
            metrics, new_support = tester.test_fewshot(test_ent2quad,
                                                       testDataset.__len__(),
                                                       baseData_backtest.skip_dict,
                                                       baseData_backtest.skip_dict_unaware,
                                                       config['num_ent'],
                                                       old_support)
            old_support = new_support
            for mode in metrics.keys():
                logging.info('Test {} : {}'.format(mode, metrics[mode]))
            if best_mrr < metrics['MRR-unaware']:
                best_mrr = metrics['MRR-unaware']
                best_epoch = i
                trainer.save_model()
                logging.info('Save Model in {}'.format(args.save_path))
            logging.info('Best MRR {} at epoch {}'.format(best_mrr, best_epoch))

if __name__ == '__main__':
    args = parse_args()
    main(args)
