import code

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, config, norm_stats, y_scaler):
        super(Model, self).__init__()
        self.config = config
        self.norm_stats = norm_stats
        self.y_scaler = y_scaler

        self.n_company = 63
        self.n_job = 9
        self.n_degree = 5
        self.n_major = 9
        self.n_industry = 8

        self.embedding_dim = config.embedding_dim
        self.hidden1_dim = config.hidden1_dim
        self.hidden2_dim = config.hidden2_dim

        if self.config.encode_type == "embedding":
            #self.company_emb = nn.Embedding(
            #    self.n_company,
            #    self.embedding_dim
            #)
            self.job_emb = nn.Embedding(
                self.n_job,
                self.embedding_dim,
                padding_idx=0,
            )
            self.degree_emb = nn.Embedding(
                self.n_degree,
                self.embedding_dim,
                padding_idx=0,
            )
            self.major_emb = nn.Embedding(
                self.n_major,
                self.embedding_dim,
                padding_idx=0,
            )
            self.industry_emb = nn.Embedding(
                self.n_industry,
                self.embedding_dim,
                padding_idx=0,
            )
            self.cont_fc = nn.Linear(
                2,
                self.embedding_dim
            )
            #self.company_emb.apply(self.init_linear_weights)
            self.job_emb.apply(self.init_linear_weights)
            self.degree_emb.apply(self.init_linear_weights)
            self.major_emb.apply(self.init_linear_weights)
            self.industry_emb.apply(self.init_linear_weights)
            self.cont_fc.apply(self.init_linear_weights)

            self.hidden_fc = nn.Sequential(
                nn.Tanh(),
                nn.Linear(self.embedding_dim*5, self.hidden1_dim),
                nn.Tanh(),
                nn.Linear(self.hidden1_dim, self.hidden2_dim),
                # nn.Tanh(),
            )
            self.hidden_fc.apply(self.init_linear_weights)
        else:
            self.feat_fc = nn.Linear(
                self.n_company+self.n_job+self.n_degree+self.n_major+self.n_industry+2,
                self.hidden1_dim,
            )
            self.feat_fc.apply(self.init_linear_weights)

            self.hidden_fc = nn.Sequential(
                nn.Tanh(),
                nn.Linear(self.hidden1_dim, self.hidden1_dim),
                nn.Tanh(),
                nn.Linear(self.hidden1_dim, self.hidden2_dim),
                # nn.Tanh(),
            )
            self.hidden_fc.apply(self.init_linear_weights)

        self.output_fc = nn.Linear(
            self.hidden2_dim,
            1
        )
        self.output_fc.apply(self.init_linear_weights)

        if self.config.optimizer == "adam":
            self.optimizer = optim.Adam(
                self.parameters(),
                lr=self.config.init_lr,
            )
        elif self.config.optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.parameters(),
                lr=self.config.init_lr,
            )

    def init_linear_weights(self, m, init_w=0.08):
        if isinstance(m, nn.Linear):
            m.weight.data.uniform_(-1.0*init_w, init_w)
            m.bias.data.fill_(0)

    def load_model(self, model_path):
        if DEVICE == "cuda":
            pretrained_state_dict = torch.load(model_path)
        else:
            pretrained_state_dict = torch.load(model_path,
                                                map_location=lambda storage, loc: storage)
        self.load_state_dict(pretrained_state_dict)

    def forward(self, input_disc, input_cont):

        if self.config.encode_type == "embedding":
            #company_embeddings = self.company_emb(input_disc[:, 0])
            job_embeddings = self.job_emb(input_disc[:, 1])
            degree_embeddings = self.degree_emb(input_disc[:, 2])
            major_embeddings = self.major_emb(input_disc[:, 3])
            industry_embeddings = self.industry_emb(input_disc[:, 4])
            cont_embeddings = self.cont_fc(input_cont)

            feats = torch.cat([
                #company_embeddings,
                job_embeddings,
                degree_embeddings,
                major_embeddings,
                industry_embeddings,
                cont_embeddings,
            ], dim=1)
        else:
            batch_size = input_disc.size(0)
            company_one_hot = torch.zeros(batch_size, self.n_company).to(DEVICE)
            company_one_hot.scatter_(1, input_disc[:, 0].view(-1, 1), 1)
            job_one_hot = torch.zeros(batch_size, self.n_job).to(DEVICE)
            job_one_hot.scatter_(1, input_disc[:, 1].view(-1, 1), 1)
            degree_one_hot = torch.zeros(batch_size, self.n_degree).to(DEVICE)
            degree_one_hot.scatter_(1, input_disc[:, 2].view(-1, 1), 1)
            major_one_hot = torch.zeros(batch_size, self.n_major).to(DEVICE)
            major_one_hot.scatter_(1, input_disc[:, 3].view(-1, 1), 1)
            industry_one_hot = torch.zeros(batch_size, self.n_industry).to(DEVICE)
            industry_one_hot.scatter_(1, input_disc[:, 4].view(-1, 1), 1)

            feats = torch.cat([
                company_one_hot,
                job_one_hot,
                degree_one_hot,
                major_one_hot,
                industry_one_hot,
                input_cont,
            ], dim=1)
            feats = self.feat_fc(feats)

        hidden = self.hidden_fc(feats)
        output = self.output_fc(hidden).view(-1)

        # code.interact(local=locals())

        return output

    def train_step(self, data, lr, step):
        self.ret_statistics = {}

        X_disc, X_cont, Y = data["X_disc"], data["X_cont"], data["Y"]

        Y_pred = self.forward(X_disc, X_cont)

        loss_func = torch.nn.MSELoss(reduction="mean")
        loss = loss_func(Y_pred, Y)
        self.ret_statistics["loss"] = loss.item()

        # update
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.gradient_clip)
        self.optimizer.step()

        # if step % 1000 == 0:
        #     code.interact(local=locals())

    def evaluate_step(self, data):
        self.ret_statistics = {}

        X_disc, X_cont, Y = data["X_disc"], data["X_cont"], data["Y"]

        Y_pred = self.forward(X_disc, X_cont)

        # rescale
        # Y_pred = Y_pred*(self.norm_stats["y"]["max"]-self.norm_stats["y"]["min"]) + self.norm_stats["y"]["min"]
        # Y = Y*(self.norm_stats["y"]["max"]-self.norm_stats["y"]["min"]) + self.norm_stats["y"]["min"]

        # inverse standarize
        if self.y_scaler is not None:
            Y = torch.FloatTensor(self.y_scaler.inverse_transform(Y.cpu())).to(DEVICE)
            Y_pred = torch.FloatTensor(self.y_scaler.inverse_transform(Y_pred.cpu().detach())).to(DEVICE)

        if Y is not None:
            loss = F.mse_loss(Y_pred, Y)
            self.ret_statistics["loss"] = loss.item()

        return Y_pred
