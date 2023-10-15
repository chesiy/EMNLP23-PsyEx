import enum
from cv2 import log
import torch
import numpy as np
import pandas as pd
from torch import nn, optim
import pytorch_lightning as pl
from transformers import AutoModel
from argparse import ArgumentParser
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from torch.nn import functional as F

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

id2disease = [
    "adhd",
    "anxiety",
    "bipolar",
    "depression",
    "eating",
    "ocd",
    "ptsd"
]

def get_avg_metrics(all_labels, all_probs, threshold, disease='None', setting='binary', class_names=id2disease):
    labels_by_class = []
    probs_by_class = []
    if disease != 'None':
        dis_id = id2disease.index(disease)
        if setting == 'binary':
            sel_indices = np.where(all_labels[:, dis_id] != -1)
            labels = all_labels[:, dis_id][sel_indices]
            probs = all_probs[:, dis_id][sel_indices]
        else:
            labels = all_labels[:, dis_id]
            probs = all_probs[:, dis_id]
        ret = {}
        preds = (probs > threshold).astype(float)
        ret["macro_acc"]=np.mean(labels == preds)
        ret["macro_p"]=precision_score(labels, preds)
        ret["macro_r"]=recall_score(labels, preds)
        ret["macro_f1"]=f1_score(labels, preds)
        try:
            ret["macro_auc"]=roc_auc_score(labels, probs)
        except:
            ret["macro_auc"]=0.5
    else:
        for i in range(all_labels.shape[1]):
            if setting == 'binary':
                sel_indices = np.where(all_labels[:, i] != -1)
                labels_by_class.append(all_labels[:, i][sel_indices])
                probs_by_class.append(all_probs[:, i][sel_indices])
            else:
                labels_by_class.append(all_labels[:, i])
                probs_by_class.append(all_probs[:, i])
        # macro avg metrics
        ret = {}
        for k in ["macro_acc", "macro_p", "macro_r", "macro_f1", "macro_auc"]:
            ret[k] = []
        for labels, probs in zip(labels_by_class, probs_by_class):
            preds = (probs > threshold).astype(float)
            ret["macro_acc"].append(np.mean(labels == preds))
            ret["macro_p"].append(precision_score(labels, preds))
            ret["macro_r"].append(recall_score(labels, preds))
            ret["macro_f1"].append(f1_score(labels, preds))
            try:
                ret["macro_auc"].append(roc_auc_score(labels, probs))
            except:
                ret["macro_auc"].append(0.5)
        for k in ["macro_acc", "macro_p", "macro_r", "macro_f1", "macro_auc"]:
            # list of diseases
            for class_name, v in zip(class_names, ret[k]):
                ret[class_name+"_"+k[6:]] = v
            ret[k] = np.mean(ret[k])

        if setting != 'binary':
            all_preds = (all_probs > threshold).astype(float)
            ret["micro_p"] = precision_score(all_labels.flatten(), all_preds.flatten())
            ret["micro_r"] = recall_score(all_labels.flatten(), all_preds.flatten())
            ret["micro_f1"] = f1_score(all_labels.flatten(), all_preds.flatten())
            ret["sample_acc"] = accuracy_score(all_labels, all_preds)

    return ret

def masked_logits_loss(logits, labels, masks=None):
    # treat unlabeled samples(-1) as implict negative (0.)
    labels2 = torch.clamp_min(labels, 0.)
    losses = F.binary_cross_entropy_with_logits(logits, labels2, reduction='none')
    if masks is not None:
        masked_losses = torch.masked_select(losses, masks)
        return masked_losses.mean()
    else:
        return losses.mean()

class LightningInterface(pl.LightningModule):
    def __init__(self, threshold=0.5, setting="binary", **kwargs):
        super().__init__()
        self.best_f1 = 0.
        self.threshold = threshold
        self.disease = kwargs['disease']
        self.criterion = masked_logits_loss
        self.setting = setting

    def training_step(self, batch, batch_nb, optimizer_idx=0):
        x, y, label_masks = batch
        y_hat = self(x)
        if type(y_hat) == tuple:
            y_hat, attn_scores = y_hat
        if self.setting == 'binary':
            loss = self.criterion(y_hat, y, label_masks)
        else:
            loss = self.criterion(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_nb):
        x, y, label_masks = batch
        y_hat = self(x)
        if type(y_hat) == tuple:
            y_hat, attn_scores = y_hat
        yy, yy_hat = y.detach().cpu().numpy(), y_hat.sigmoid().detach().cpu().numpy()
        if self.setting == 'binary':
            loss = self.criterion(y_hat, y, label_masks)
        else:
            loss = self.criterion(y_hat, y)
        return {'val_loss': loss, "labels": yy, "probs": yy_hat}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        all_labels = np.concatenate([x['labels'] for x in outputs])
        all_probs = np.concatenate([x['probs'] for x in outputs])
        ret = get_avg_metrics(all_labels, all_probs, self.threshold, self.disease, self.setting)
        print('val res', ret)
        if self.current_epoch == 0:  # prevent the initial check modifying it
            self.best_f1 = 0
        self.best_f1 = max(self.best_f1, ret['macro_f1'])
        tensorboard_logs = {'val_loss': avg_loss, 'hp_metric': self.best_f1, 'val_f1': ret['macro_f1']}
        self.log_dict(tensorboard_logs)
        self.log("best_f1", self.best_f1, prog_bar=True, on_epoch=True)
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        x, y, label_masks = batch
        y_hat = self(x)
        if type(y_hat) == tuple:
            y_hat, attn_scores = y_hat
        yy, yy_hat = y.detach().cpu().numpy(), y_hat.sigmoid().detach().cpu().numpy()
        if self.setting == 'binary':
            loss = self.criterion(y_hat, y, label_masks)
        else:
            loss = self.criterion(y_hat, y)
        return {'test_loss': loss, "labels": yy, "probs": yy_hat}
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        all_labels = np.concatenate([x['labels'] for x in outputs])
        all_probs = np.concatenate([x['probs'] for x in outputs])
        ret = get_avg_metrics(all_labels, all_probs, self.threshold, self.disease, self.setting)
        results = {'test_loss': avg_loss}
        for k, v in ret.items():
            results[f"test_{k}"] = v
        self.log_dict(results)
        return results

    def on_after_backward(self):
        pass
        # can check gradient
        # global_step = self.global_step
        # if int(global_step) % 100 == 0:
        #     for name, param in self.named_parameters():
        #         self.logger.experiment.add_histogram(name, param, global_step)
        #         if param.requires_grad:
        #             self.logger.experiment.add_histogram(f"{name}_grad", param.grad, global_step)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        return parser


class PsyEx_wo_symp(nn.Module):
    def __init__(self, model_type, num_heads=8, num_trans_layers=6, max_posts=32, freeze=False, pool_type="first") -> None:
        super().__init__()
        self.model_type = model_type
        self.num_heads = num_heads
        self.num_trans_layers = num_trans_layers
        self.pool_type = pool_type
        self.post_encoder = AutoModel.from_pretrained(model_type)
        if freeze:
            for name, param in self.post_encoder.named_parameters():
                param.requires_grad = False
        self.hidden_dim = self.post_encoder.config.hidden_size
        self.max_posts = max_posts
        self.pos_emb = nn.Parameter(torch.Tensor(max_posts, self.hidden_dim))
        nn.init.xavier_uniform_(self.pos_emb)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, dim_feedforward=self.hidden_dim, nhead=num_heads, activation='gelu')
        self.user_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_trans_layers)
        self.attn_ff = nn.ModuleList([nn.Linear(self.hidden_dim, 1) for disease in id2disease])
        self.dropout = nn.Dropout(self.post_encoder.config.hidden_dropout_prob)
        self.clf = nn.ModuleList([nn.Linear(self.hidden_dim, 1) for disease in id2disease])
    
    def forward(self, batch, **kwargs):
        feats = []
        attn_scores = []
        for user_feats in batch:
            post_outputs = self.post_encoder(user_feats["input_ids"], user_feats["attention_mask"], user_feats["token_type_ids"])
            # [num_posts, seq_len, hidden_size] -> [num_posts, 1, hidden_size]
            if self.pool_type == "first":
                x = post_outputs.last_hidden_state[:, 0:1, :]
            elif self.pool_type == 'mean':
                x = mean_pooling(post_outputs.last_hidden_state, user_feats["attention_mask"]).unsqueeze(1)
            # positional embedding for posts
            x = x + self.pos_emb[:x.shape[0], :].unsqueeze(1)
            x = self.user_encoder(x).squeeze(1) # [num_posts, hidden_size]
            attn_score = [torch.softmax(attn_ff(x).squeeze(), -1) for attn_ff in self.attn_ff]
            # weighted sum [hidden_size, ]
            feat = [self.dropout(score @ x) for score in attn_score]
            feats.append(feat)
            attn_scores.append(attn_score)

        logits = []
        for i in range(len(id2disease)):
            tmp = [feats[j][i] for j in range(len(feats))]
            logit = self.clf[i](torch.stack(tmp))
            logits.append(logit)
        logits = torch.stack(logits, dim=0).transpose(0, 1).squeeze()
        return logits, attn_scores


class TwoStreamPsyEx(nn.Module):
    def __init__(self, model_type, num_heads=8, num_trans_layers=6, max_posts=64, freeze=False, pool_type="first") -> None:
        super().__init__()
        self.model_type = model_type
        self.num_heads = num_heads
        self.num_trans_layers = num_trans_layers
        self.pool_type = pool_type
        self.post_encoder = AutoModel.from_pretrained(model_type)
        if freeze:
            for name, param in self.post_encoder.named_parameters():
                param.requires_grad = False
        self.hidden_dim = self.post_encoder.config.hidden_size
        self.max_posts = max_posts
        self.pos_emb = nn.Parameter(torch.Tensor(max_posts, self.hidden_dim))
        nn.init.xavier_uniform_(self.pos_emb)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, dim_feedforward=self.hidden_dim, nhead=num_heads, activation='gelu')
        self.user_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_trans_layers)
        self.attn_ff = nn.ModuleList([nn.Linear(self.hidden_dim, 1) for disease in id2disease])
        self.attn_ff_for_symp = nn.ModuleList([nn.Linear(38, 1) for disease in id2disease])
        self.dropout = nn.Dropout(self.post_encoder.config.hidden_dropout_prob)
        self.clf = nn.ModuleList([nn.Linear(self.hidden_dim+38, 1) for disease in id2disease])
    
    def forward(self, batch, **kwargs):
        feats = []
        symp_feats = []
        attn_scores = []
        symp_attn_scores = []
        for user_feats in batch:
            post_outputs = self.post_encoder(user_feats["input_ids"], user_feats["attention_mask"], user_feats["token_type_ids"])
            # [num_posts, seq_len, hidden_size] -> [num_posts, 1, hidden_size]
            if self.pool_type == "first":
                x = post_outputs.last_hidden_state[:, 0:1, :]
            elif self.pool_type == 'mean':
                x = mean_pooling(post_outputs.last_hidden_state, user_feats["attention_mask"]).unsqueeze(1)
            # positional embedding for posts
            x = x + self.pos_emb[:x.shape[0], :].unsqueeze(1)
            x = self.user_encoder(x).squeeze(1) # [num_posts, hidden_size]
            attn_score = [torch.softmax(attn_ff(x).squeeze(), -1) for attn_ff in self.attn_ff]
            symp_attn_score = [torch.softmax(attn_ff(user_feats["symp"]).squeeze(), -1) for attn_ff in self.attn_ff_for_symp]
            # weighted sum [hidden_size, ]
            feat = [self.dropout(score @ x) for score in attn_score]
            symp_feat = [self.dropout(score @ user_feats["symp"]) for score in symp_attn_score]
            feats.append(feat)
            symp_feats.append(symp_feat)
            attn_scores.append(attn_score)
            symp_attn_scores.append(symp_attn_score)

        logits = []
        user_representations = []
        for i in range(len(id2disease)):
            tmp = torch.stack([feats[j][i] for j in range(len(feats))])
            tmp_symp = torch.stack([symp_feats[j][i] for j in range(len(symp_feats))])
            logit = self.clf[i](torch.concat([tmp,tmp_symp], axis=1))
            user_representations.append(torch.concat([tmp,tmp_symp], axis=1))
            logits.append(logit)
        logits = torch.stack(logits, dim=0).transpose(0, 1).squeeze()
        user_representations = torch.stack(user_representations, dim=0).transpose(0, 1).squeeze()
        return logits, (attn_scores, symp_attn_scores, user_representations)


class PsyEx_wo_multi_attn(nn.Module):
    '''with absolute learned positional embedding for post level'''
    def __init__(self, model_type, num_heads=8, num_trans_layers=6, max_posts=64, freeze=False, pool_type="first") -> None:
        super().__init__()
        self.model_type = model_type
        self.num_heads = num_heads
        self.num_trans_layers = num_trans_layers
        self.pool_type = pool_type
        self.post_encoder = AutoModel.from_pretrained(model_type)
        if freeze:
            for name, param in self.post_encoder.named_parameters():
                param.requires_grad = False
        self.hidden_dim = self.post_encoder.config.hidden_size
        self.max_posts = max_posts
        self.pos_emb = nn.Parameter(torch.Tensor(max_posts, self.hidden_dim))
        nn.init.xavier_uniform_(self.pos_emb)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, dim_feedforward=self.hidden_dim, nhead=num_heads, activation='gelu')
        self.user_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_trans_layers)
        self.attn_ff = nn.Linear(self.hidden_dim, 1)
        self.attn_ff_for_symp = nn.Linear(38, 1)
        self.dropout = nn.Dropout(self.post_encoder.config.hidden_dropout_prob)
        self.clf = nn.Linear(self.hidden_dim+38, 7)
    
    def forward(self, batch, **kwargs):
        feats = []
        symp_feats = []
        attn_scores = []
        symp_attn_scores = []
        for user_feats in batch:
            post_outputs = self.post_encoder(user_feats["input_ids"], user_feats["attention_mask"], user_feats["token_type_ids"])
            # [num_posts, seq_len, hidden_size] -> [num_posts, 1, hidden_size]
            if self.pool_type == "first":
                x = post_outputs.last_hidden_state[:, 0:1, :]
            elif self.pool_type == 'mean':
                x = mean_pooling(post_outputs.last_hidden_state, user_feats["attention_mask"]).unsqueeze(1)
            # positional embedding for posts
            x = x + self.pos_emb[:x.shape[0], :].unsqueeze(1)
            x = self.user_encoder(x).squeeze(1) # [num_posts, hidden_size]
            attn_score = torch.softmax(self.attn_ff(x).squeeze(), -1)
            symp_attn_score = torch.softmax(self.attn_ff_for_symp(user_feats["symp"]).squeeze(), -1)
            # weighted sum [hidden_size, ]
            feat = self.dropout(attn_score @ x)
            symp_feat = self.dropout(symp_attn_score @ user_feats["symp"])
            feats.append(feat)
            symp_feats.append(symp_feat)
            attn_scores.append(attn_score)
            symp_attn_scores.append(symp_attn_score)

        feats = torch.stack(feats)
        symp_feats = torch.stack(symp_feats)
        x = torch.concat([feats, symp_feats], axis=1)
        logits = self.clf(x)

        return logits, attn_scores


class HierClassifier(LightningInterface):
    def __init__(self, threshold=0.5, lr=5e-5, model_type="prajjwal1/bert-tiny", user_encoder="none", num_heads=8, num_trans_layers=2, freeze_word_level=False, pool_type="first", cnn_lr=0.003, setting="binary", **kwargs):
        super().__init__(threshold=threshold, setting=setting, **kwargs)
        self.model_type = model_type
        self.user_encoder = user_encoder
        if user_encoder == "wo_symp_stream":
            self.model = PsyEx_wo_symp(model_type, num_heads, num_trans_layers, freeze=freeze_word_level, pool_type=pool_type)
        elif user_encoder == "psyex":
            self.model = TwoStreamPsyEx(model_type, num_heads, num_trans_layers, freeze=freeze_word_level, pool_type=pool_type)
        elif user_encoder == "wo_multi_attn":
            self.model = PsyEx_wo_multi_attn(model_type, num_heads, num_trans_layers, freeze=freeze_word_level, pool_type=pool_type)
        self.lr = lr
        self.cnn_lr = cnn_lr
        self.save_hyperparameters()
        print(self.hparams)

    def forward(self, x):
        x = self.model(x)
        return x

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = LightningInterface.add_model_specific_args(parser)
        parser.add_argument("--threshold", type=float, default=0.5)
        parser.add_argument("--lr", type=float, default=2e-4)
        parser.add_argument("--cnn_lr", type=float, default=0.003)
        parser.add_argument("--user_encoder", type=str, default="none")
        parser.add_argument("--pool_type", type=str, default="first")
        parser.add_argument("--num_heads", type=int, default=8)
        parser.add_argument("--num_trans_layers", type=int, default=2)
        parser.add_argument("--freeze_word_level", action="store_true")
        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer