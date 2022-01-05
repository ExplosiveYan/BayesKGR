import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from layers import GCNConv, GATConv, RGCNConv, GENConv

class InducGEN(nn.Module):

    def __init__(self, entity_embedding_dim, relation_embedding_dim, num_entities, num_relations, args, entity_embedding, relation_embedding, entity_var=None, relation_var=None):

        super(InducGEN, self).__init__()

        self.args = args

        self.entity_embedding_dim = entity_embedding_dim
        self.relation_embedding_dim = relation_embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        if self.args.pre_train_model == 'KG2E':
            self.entity_embedding = nn.Embedding(num_entities, entity_embedding_dim)
            self.entity_covar = nn.Embedding(num_entities, entity_embedding_dim)
            self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, relation_embedding_dim))
            self.relation_covar = nn.Parameter(torch.Tensor(num_relations, relation_embedding_dim))

        else:
            self.entity_embedding = nn.Embedding(num_entities, entity_embedding_dim)
            self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, relation_embedding_dim))

        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))

        if self.args.pre_train:
            if self.args.pre_train_model == 'KG2E':
                print(entity_embedding.shape, relation_embedding.shape)

                self.entity_embedding.weight.data.copy_(entity_embedding[:num_entities].clone().detach())
                self.entity_covar.weight.data.copy_(entity_embedding[num_entities::].clone().detach())
                self.relation_embedding.data.copy_(relation_embedding[:num_relations].clone().detach())
                self.relation_covar.data.copy_(relation_embedding[num_relations::].clone().detach())

                self.entity_embedding.weight.data.copy_(torch.renorm(input=self.entity_embedding.weight.data.detach().cpu(),
                                                                    p=2,
                                                                    dim=0,
                                                                    maxnorm=1.0))
                self.relation_embedding.data.copy_(
                    torch.renorm(input=self.relation_embedding.data.detach().cpu(),
                                 p=2,
                                 dim=0,
                                 maxnorm=1.0))
                self.entity_covar.weight.data.copy_(torch.clamp(input=self.entity_covar.weight.data.detach().cpu(),
                                                               min=0.03,
                                                               max=3.0))
                self.relation_covar.data.copy_(torch.clamp(input=self.relation_covar.data.detach().cpu(),
                                                                 min=0.03,
                                                                 max=3.0))


            else:
                self.entity_embedding.weight.data.copy_(entity_embedding.clone().detach())
                self.relation_embedding.data.copy_(relation_embedding.clone().detach())

            if not self.args.fine_tune:
                self.entity_embedding.weight.requires_grad = False
                self.relation_embedding.requires_grad = False

        self.dropout = nn.Dropout(args.dropout)

        self.gnn = GENConv(self.entity_embedding_dim + self.relation_embedding_dim, self.entity_embedding_dim, self.num_relations * 2, num_bases = self.args.bases, root_weight = False, bias = False)
        if self.args.pre_train_model == 'KG2E':
            self.gnn_var = GENConv(self.entity_embedding_dim + self.relation_embedding_dim, self.entity_embedding_dim, self.num_relations * 2, num_bases = self.args.bases, root_weight = False, bias = False)
        self.score_function = self.args.score_function

    def forward(self, unseen_entity, triplets, use_cuda):

        # Pre-process
        src, rel, dst = triplets.transpose()
        uniq_v, edges = np.unique((src, dst), return_inverse=True)
        src, dst = np.reshape(edges, (2, -1))

        unseen_index = np.where(uniq_v == unseen_entity)[0][0]
        rel_index = np.concatenate((rel, rel))

        src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
        rel = np.concatenate((rel, rel + self.num_relations))

        # Torch
        node_id = torch.LongTensor(uniq_v)
        edge_index = torch.stack((
            torch.LongTensor(src),
            torch.LongTensor(dst)
        ))
        edge_type = torch.LongTensor(rel)

        if use_cuda:
            node_id = node_id.cuda()
            edge_index = edge_index.cuda()
            edge_type = edge_type.cuda()
        if self.args.pre_train_model == 'KG2E':
            x = self.entity_embedding(node_id)
            y = self.entity_covar(node_id)
            rel_emb = self.relation_embedding[rel_index]
            rel_var = self.relation_covar[rel_index]
            print("x = entity_embedding", x)
            # print(node_id, edge_index, edge_type)
            # print("rel_emb", rel_emb)
            # print("rel_var", rel_var)
            print("GNN parameters")
            print(x.shape, y.shape, edge_index.shape, edge_index.shape, rel_emb.shape)
            embeddings = self.gnn(x, edge_index, edge_type, rel_emb, edge_norm = None)
            covar = self.gnn_var(y, edge_index, edge_type, rel_var, edge_norm = None)
            unseen_entity_embedding = embeddings[unseen_index]
            unseen_entity_embedding = self.dropout(unseen_entity_embedding)
            # unseen_entity_embedding = torch.unsqueeze(unseen_entity_embedding, 1)
            unseen_entity_covar = covar[unseen_index]
            unseen_entity_covar = self.dropout(unseen_entity_covar)
            # unseen_entity_covar = torch.unsqueeze(unseen_entity_covar, 1)

            unseen_entity_embedding = torch.cat([unseen_entity_embedding, unseen_entity_covar], 0)
            # print(unseen_entity_embedding.shape)
            return unseen_entity_embedding

        else:
            x = self.entity_embedding(node_id)
            rel_emb = self.relation_embedding[rel_index]


            embeddings = self.gnn(x, edge_index, edge_type, rel_emb, edge_norm=None)

            unseen_entity_embedding = embeddings[unseen_index]
            unseen_entity_embedding = self.dropout(unseen_entity_embedding)

            return unseen_entity_embedding

    def score_loss(self, unseen_entity, unseen_entity_embedding, triplets, target, use_cuda):

        if self.args.pre_train_model == 'KG2E':
            head_embeddings = self.entity_embedding(triplets[:, 0])
            head_covar = self.entity_covar(triplets[:, 0])
            relation_embeddings = self.relation_embedding[triplets[:, 1]]
            relation_covars = self.relation_covar[triplets[:, 1]]
            tail_embeddings = self.entity_embedding(triplets[:, 2])
            tail_covar = self.entity_covar(triplets[:, 2])

            head_embeddings[triplets[:, 0] == unseen_entity] = unseen_entity_embedding[:self.entity_embedding_dim]
            head_covar[triplets[:, 0] == unseen_entity] = unseen_entity_embedding[self.entity_embedding_dim:]
            tail_embeddings[triplets[:, 2] == unseen_entity] = unseen_entity_embedding[:self.entity_embedding_dim]
            tail_covar[triplets[:, 2] == unseen_entity] = unseen_entity_embedding[self.entity_embedding_dim:]
        else:
            head_embeddings = self.entity_embedding(triplets[:, 0])
            relation_embeddings = self.relation_embedding[triplets[:, 1]]
            tail_embeddings = self.entity_embedding(triplets[:, 2])

            head_embeddings[triplets[:, 0] == unseen_entity] = unseen_entity_embedding
            tail_embeddings[triplets[:, 2] == unseen_entity] = unseen_entity_embedding

        len_positive_triplets = int(len(target) / (self.args.negative_sample + 1))
        
        if self.score_function == 'DistMult':

            score = head_embeddings * relation_embeddings * tail_embeddings
            score = torch.sum(score, dim = 1)

            positive_score = score[:len_positive_triplets]
            negative_score = score[len_positive_triplets:]
            # print("negative_score:{}\n positive_score:{}".format(positive_score, negative_score))


        elif self.score_function == 'TransE':

            pos_head_embeddings = head_embeddings[:len_positive_triplets]
            pos_relation_embeddings = relation_embeddings[:len_positive_triplets]
            pos_tail_embeddings = tail_embeddings[:len_positive_triplets]

            x = pos_head_embeddings + pos_relation_embeddings - pos_tail_embeddings
            positive_score = - torch.norm(x, p = 2, dim = 1)

            neg_head_embeddings = head_embeddings[len_positive_triplets:]
            neg_relation_embeddings = relation_embeddings[len_positive_triplets:]
            neg_tail_embeddings = tail_embeddings[len_positive_triplets:]

            x = neg_head_embeddings + neg_relation_embeddings - neg_tail_embeddings
            # print("x = ", x.shape)

            negative_score = - torch.norm(x, p = 2, dim = 1)


        # TODO: 添加KG2E的score_function, 需要考虑到 entity_embeddings, rel_embedding, entity_var, rel_var, calculate the score.

        elif self.score_function == 'KG2E':

            def KLScore(**kwargs):
                # Calculate KL(e, r)
                losep1 = torch.sum(kwargs["errorv"] / kwargs["relationv"], dim=1)
                losep2 = torch.sum((kwargs["relationm"] - kwargs["errorm"]) ** 2 / kwargs["relationv"], dim=1)
                KLer = (losep1 + losep2 - self.entity_embedding_dim) / 2

                # Calculate KL(r, e)
                losep1 = torch.sum(kwargs["relationv"] / kwargs["errorv"], dim=1)
                losep2 = torch.sum((kwargs["errorm"] - kwargs["relationm"]) ** 2 / kwargs["errorv"], dim=1)
                KLre = (losep1 + losep2 - self.entity_embedding_dim) / 2
                x = (KLer + KLre) / 2
                # print("x = ", x.shape)
                return x

            pos_head_embeddings = head_embeddings[:len_positive_triplets]
            pos_head_covar = head_covar[:len_positive_triplets]
            pos_relation_embeddings = relation_embeddings[:len_positive_triplets]
            pos_relation_covar = relation_covars[:len_positive_triplets]
            pos_tail_embeddings = tail_embeddings[:len_positive_triplets]
            pos_tail_covar = tail_covar[:len_positive_triplets]
            pos_errorm = pos_tail_embeddings - pos_head_embeddings
            pos_errorv = pos_tail_covar - pos_head_covar

            neg_head_embeddings = head_embeddings[len_positive_triplets:]
            neg_head_covar = head_covar[len_positive_triplets:]
            neg_relation_embeddings = relation_embeddings[len_positive_triplets:]
            neg_relation_covar = relation_covars[len_positive_triplets:]
            neg_tail_embeddings = tail_embeddings[len_positive_triplets:]
            neg_tail_covar = tail_covar[len_positive_triplets:]
            neg_errorm = neg_tail_embeddings - neg_head_embeddings
            neg_errorv = neg_tail_covar - neg_head_covar
            positive_score = KLScore(relationm=pos_relation_embeddings, relationv=pos_relation_covar, errorm=pos_errorm, errorv=pos_errorv)
            negative_score = KLScore(relationm=neg_relation_embeddings, relationv=neg_relation_covar, errorm=neg_errorm, errorv=neg_errorv)
            # print("negative_score:{}\n positive_score:{}".format(positive_score, negative_score))
        else:

            raise ValueError("Score Function Name <{}> is Wrong".format(self.score_function))

        y = torch.ones(len_positive_triplets * self.args.negative_sample)

        if use_cuda:
            y = y.cuda()

        positive_score = positive_score.repeat(self.args.negative_sample)
        # print("neg score = ", negative_score)
        # print("y = ", y)
        # print("positive score = ", positive_score)
        loss = F.margin_ranking_loss(positive_score, negative_score, y, margin = self.args.margin)
        # print("loss = ", loss)
        return loss


class TransGEN(nn.Module):

    def __init__(self, entity_embedding_dim, relation_embedding_dim, num_entities, num_relations, args, entity_embedding, relation_embedding):

        super(TransGEN, self).__init__()

        self.args = args

        self.entity_embedding_dim = entity_embedding_dim
        self.relation_embedding_dim = relation_embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations

        self.entity_embedding = nn.Embedding(num_entities, entity_embedding_dim)
        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, relation_embedding_dim))

        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))

        if self.args.pre_train:
            self.entity_embedding.weight.data.copy_(entity_embedding.clone().detach())
            self.relation_embedding.data.copy_(relation_embedding.clone().detach())

            if not self.args.fine_tune:
                self.entity_embedding.weight.requires_grad = False
                self.relation_embedding.requires_grad = False

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

        self.gnn_induc = GENConv(self.entity_embedding_dim + self.relation_embedding_dim, self.entity_embedding_dim, self.num_relations * 2, num_bases = self.args.bases, root_weight = False, bias = False)
        self.gnn_trans_mu = GENConv(self.entity_embedding_dim + self.relation_embedding_dim, self.entity_embedding_dim, self.num_relations * 2, num_bases = self.args.bases)
        self.gnn_trans_sigma = GENConv(self.entity_embedding_dim + self.relation_embedding_dim, self.entity_embedding_dim, self.num_relations * 2, num_bases = self.args.bases)
        
        self.score_function = self.args.score_function

    def reparameterize(self, mu, logvar):

        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps*std
        else:
            return mu

    def forward(self, unseen_entity, triplets, use_cuda, is_trans = True, total_unseen_entity_embedding = None):

        if is_trans == False:

            # Pre-process
            src, rel, dst = triplets.transpose()
            uniq_v, edges = np.unique((src, dst), return_inverse=True)
            src, dst = np.reshape(edges, (2, -1))

            unseen_index = np.where(uniq_v == unseen_entity)[0][0]
            rel_index = np.concatenate((rel, rel))

            src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
            rel = np.concatenate((rel, rel + self.num_relations))

            # Torch
            node_id = torch.LongTensor(uniq_v)
            edge_index = torch.stack((
                torch.LongTensor(src),
                torch.LongTensor(dst)
            ))
            edge_type = torch.LongTensor(rel)

            if use_cuda:
                node_id = node_id.cuda()
                edge_index = edge_index.cuda()
                edge_type = edge_type.cuda()

            x = self.entity_embedding(node_id)
            rel_emb = self.relation_embedding[rel_index]

            embeddings = self.gnn_induc(x, edge_index, edge_type, rel_emb, edge_norm = None)
            unseen_entity_embedding = embeddings[unseen_index]
            unseen_entity_embedding = self.dropout(self.relu(unseen_entity_embedding))

            return unseen_entity_embedding

        else:

            src, rel, dst = triplets.transpose()
            uniq_v, edges = np.unique((src, dst), return_inverse=True)
            src, dst = np.reshape(edges, (2, -1))

            unseen_index = []
            for entity in unseen_entity:
                unseen_index.append(np.where(uniq_v == entity)[0][0])
            rel_index = np.concatenate((rel, rel))

            src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
            rel = np.concatenate((rel, rel + self.num_relations))

            # Torch
            node_id = torch.LongTensor(uniq_v)
            edge_index = torch.stack((
                torch.LongTensor(src),
                torch.LongTensor(dst)
            ))
            edge_type = torch.LongTensor(rel)

            if use_cuda:
                node_id = node_id.cuda()
                edge_index = edge_index.cuda()
                edge_type = edge_type.cuda()

            x = self.entity_embedding(node_id)
            rel_emb = self.relation_embedding[rel_index]

            x[unseen_index] = total_unseen_entity_embedding

            mu = self.gnn_trans_mu(x, edge_index, edge_type, rel_emb, edge_norm = None)
            logvar = self.gnn_trans_sigma(x, edge_index, edge_type, rel_emb, edge_norm = None)
            embeddings = self.reparameterize(mu, logvar)

            unseen_entity_embedding = embeddings[unseen_index]
            unseen_entity_embedding = self.dropout(unseen_entity_embedding)

            return unseen_entity_embedding, mu[unseen_index], logvar[unseen_index]

    def score_loss(self, unseen_entity, unseen_entity_embedding, triplets, target, use_cuda):

        head_embeddings = self.entity_embedding(triplets[:, 0])
        relation_embeddings = self.relation_embedding[triplets[:, 1]]
        tail_embeddings = self.entity_embedding(triplets[:, 2])

        for index, entity in enumerate(unseen_entity):
            head_embeddings[triplets[:, 0] == entity] = unseen_entity_embedding[index]
            tail_embeddings[triplets[:, 2] == entity] = unseen_entity_embedding[index]
            
        len_positive_triplets = int(len(triplets) / (self.args.negative_sample + 1))

        if self.score_function == 'DistMult':

            score = head_embeddings * relation_embeddings * tail_embeddings
            score = torch.sum(score, dim = 1)

            positive_score = score[:len_positive_triplets]
            negative_score = score[len_positive_triplets:]

        elif self.score_function == 'TransE':

            pos_head_embeddings = head_embeddings[:len_positive_triplets]
            pos_relation_embeddings = relation_embeddings[:len_positive_triplets]
            pos_tail_embeddings = tail_embeddings[:len_positive_triplets]

            x = pos_head_embeddings + pos_relation_embeddings - pos_tail_embeddings
            positive_score = - torch.norm(x, p = 2, dim = 1)

            neg_head_embeddings = head_embeddings[len_positive_triplets:]
            neg_relation_embeddings = relation_embeddings[len_positive_triplets:]
            neg_tail_embeddings = tail_embeddings[len_positive_triplets:]

            x = neg_head_embeddings + neg_relation_embeddings - neg_tail_embeddings
            negative_score = - torch.norm(x, p = 2, dim = 1)

        else:

            raise ValueError("Score Function Name <{}> is Wrong".format(self.score_function))

        y = torch.ones(len_positive_triplets * self.args.negative_sample)

        if use_cuda:
            y = y.cuda()

        positive_score = positive_score.repeat(self.args.negative_sample)

        loss = F.margin_ranking_loss(positive_score, negative_score, y, margin = self.args.margin)

        return loss