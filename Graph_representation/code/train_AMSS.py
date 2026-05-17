# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
import data_utils
from numpy import loadtxt

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0]))

# ========================= 配置参数 =========================
herb_num = 563
gene_num = 2106
factor_num = 64
batch_size = 2048
top_k = 20
run_id = "1"


amss_enabled = True
amss_update_ratio = 0.3
amss_importance_momentum = 0.9
amss_importance_type = "squared_grad"

print(f"Run ID: {run_id}")
print(f"AMSS Enabled: {amss_enabled}")
print(f"AMSS Update Ratio: {amss_update_ratio}")
print(f"AMSS Importance Type: {amss_importance_type}")


path_save_model_base = './model/' + '/s_AMSS' + run_id
if (os.path.exists(path_save_model_base) == False):
    os.makedirs(path_save_model_base)


print("Loading data...")
training_herb_set, training_gene_set, training_set_count = np.load('../data/training_set.npy', allow_pickle=True)
testing_herb_set, testing_gene_set, testing_set_count = np.load('../data/testing_set.npy', allow_pickle=True)
val_herb_set, val_gene_set, val_set_count = np.load('../data/val_set.npy', allow_pickle=True)
herb_rating_set_all = np.load('../data/set_all.npy', allow_pickle=True).item()

h_d = data_utils.readD(training_herb_set, herb_num)
g_d = data_utils.readD(training_gene_set, gene_num)
d_i_train = h_d
d_j_train = g_d
sparse_u_i = data_utils.readTrainSparseMatrix(training_herb_set, h_d, g_d, True)
sparse_i_u = data_utils.readTrainSparseMatrix(training_gene_set, h_d, g_d, False)

print("Data loaded successfully!")


class BPR(nn.Module):
    def __init__(self, herb_num, gene_num, factor_num, herb_gene_matrix, gene_herb_matrix, d_i_train, d_j_train):

        super(BPR, self).__init__()
        self.herb_gene_matrix = herb_gene_matrix
        self.gene_herb_matrix = gene_herb_matrix
        self.embed_herb = nn.Embedding(herb_num, factor_num)
        self.embed_gene = nn.Embedding(gene_num, factor_num)

        # load the embeddings of the component 1

        self.embed_herb.weight.data.copy_(torch.from_numpy(loadtxt("../../Knowledge_embedding/herb_LINE.txt")))
        self.embed_gene.weight.data.copy_(torch.from_numpy(loadtxt("../../Knowledge_embedding/gene_LINE.txt")))


        for i in range(len(d_i_train)):
            d_i_train[i] = [d_i_train[i]]
        for i in range(len(d_j_train)):
            d_j_train[i] = [d_j_train[i]]

        self.d_i_train = torch.cuda.FloatTensor(d_i_train)
        self.d_j_train = torch.cuda.FloatTensor(d_j_train)
        self.d_i_train = self.d_i_train.expand(-1, factor_num)
        self.d_j_train = self.d_j_train.expand(-1, factor_num)

        nn.init.normal_(self.embed_herb.weight, std=0.01)
        nn.init.normal_(self.embed_gene.weight, std=0.01)

    def forward(self, herb, gene_i, gene_j):
        herbs_embedding = self.embed_herb.weight
        genes_embedding = self.embed_gene.weight

        gcn1_herbs_embedding = (
                torch.sparse.mm(self.herb_gene_matrix, genes_embedding) + herbs_embedding.mul(self.d_i_train))
        gcn1_genes_embedding = (
                torch.sparse.mm(self.gene_herb_matrix, herbs_embedding) + genes_embedding.mul(self.d_j_train))

        gcn2_herbs_embedding = (torch.sparse.mm(self.herb_gene_matrix, gcn1_genes_embedding) + gcn1_herbs_embedding.mul(
            self.d_i_train))
        gcn2_genes_embedding = (torch.sparse.mm(self.gene_herb_matrix, gcn1_herbs_embedding) + gcn1_genes_embedding.mul(
            self.d_j_train))

        gcn3_herbs_embedding = (torch.sparse.mm(self.herb_gene_matrix, gcn2_genes_embedding) + gcn2_herbs_embedding.mul(
            self.d_i_train))
        gcn3_genes_embedding = (torch.sparse.mm(self.gene_herb_matrix, gcn2_herbs_embedding) + gcn2_genes_embedding.mul(
            self.d_j_train))

        gcn_herbs_embedding = torch.cat(
            (herbs_embedding, gcn1_herbs_embedding, gcn2_herbs_embedding, gcn3_herbs_embedding), -1)
        gcn_genes_embedding = torch.cat(
            (genes_embedding, gcn1_genes_embedding, gcn2_genes_embedding, gcn3_genes_embedding), -1)

        herb = F.embedding(herb, gcn_herbs_embedding)
        gene_i = F.embedding(gene_i, gcn_genes_embedding)
        gene_j = F.embedding(gene_j, gcn_genes_embedding)

        prediction_i = (herb * gene_i).sum(dim=-1)
        prediction_j = (herb * gene_j).sum(dim=-1)
        l2_regulization = 0.01 * (herb ** 2 + gene_i ** 2 + gene_j ** 2).sum(dim=-1)
        loss = -((prediction_i - prediction_j)).sigmoid().log().mean() + l2_regulization.mean()

        return prediction_i, prediction_j, loss



def apply_amss_masking(model, param_importances, device='cuda'):




    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:

            if amss_importance_type == "squared_grad":
                current_importance = param.grad.data.pow(2)
            elif amss_importance_type == "abs_grad":
                current_importance = param.grad.data.abs()
            else:
                raise ValueError(f"Unknown amss_importance_type: {amss_importance_type}")

            if name not in param_importances:
                param_importances[name] = torch.zeros_like(param.data, device=device)

            param_importances[name].mul_(amss_importance_momentum).add_(
                current_importance, alpha=(1 - amss_importance_momentum)
            )


            num_elements = param.numel()
            num_to_update = int(num_elements * amss_update_ratio)

            if num_to_update == 0 and num_elements > 0:
                num_to_update = 1

            if num_to_update > 0:

                flat_importance = param_importances[name].view(-1)


                _, topk_indices = torch.topk(flat_importance, num_to_update, largest=True, sorted=False)


                mask = torch.zeros_like(flat_importance, device=device)
                mask[topk_indices] = 1.0
                mask = mask.view(param.shape)


                param.grad.data.mul_(mask)
                total_params_masked += num_to_update
            else:

                param.grad.data.zero_()

    return total_params_masked


print("Creating data loaders...")
train_dataset = data_utils.BPRData(train_dict=training_herb_set, num_gene=gene_num, num_ng=5, is_training=True,
                                   data_set_count=training_set_count, all_rating=herb_rating_set_all)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

testing_dataset_loss = data_utils.BPRData(train_dict=testing_herb_set, num_gene=gene_num, num_ng=5, is_training=True,
                                          data_set_count=testing_set_count, all_rating=herb_rating_set_all)
testing_loader_loss = DataLoader(testing_dataset_loss, batch_size=batch_size, shuffle=False, num_workers=0)

val_dataset_loss = data_utils.BPRData(train_dict=val_herb_set, num_gene=gene_num, num_ng=5, is_training=True,
                                      data_set_count=val_set_count, all_rating=herb_rating_set_all)
val_loader_loss = DataLoader(val_dataset_loss, batch_size=batch_size, shuffle=False, num_workers=0)

print("Data loaders created!")


print("Initializing model...")
model = BPR(herb_num, gene_num, factor_num, sparse_u_i, sparse_i_u, d_i_train, d_j_train)
model = model.to('cuda')
optimizer_bpr = torch.optim.Adam(model.parameters(), lr=0.005)


param_importances = {}
if amss_enabled:
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_importances[name] = torch.zeros_like(param.data, device='cuda')
    print("AMSS parameter importances initialized!")

print("Model initialized!")


print('-------training with AMSS-------')
count, best_hr = 0, 0

for epoch in range(350):
    model.train()
    start_time = time.time()
    train_loader.dataset.ng_sample()
    train_loss_sum = []
    total_params_masked_epoch = 0
    batch_count = 0

    for herb, gene_i, gene_j in train_loader:
        herb = herb.cuda()
        gene_i = gene_i.cuda()
        gene_j = gene_j.cuda()

        model.zero_grad()
        prediction_i, prediction_j, loss = model(herb, gene_i, gene_j)
        loss.backward()


        if amss_enabled:
            params_masked = apply_amss_masking(model, param_importances, device='cuda')
            total_params_masked_epoch += params_masked


        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=8)

        optimizer_bpr.step()
        count += 1
        train_loss_sum.append(loss.item())
        batch_count += 1

    elapsed_time = time.time() - start_time
    train_loss = round(np.mean(train_loss_sum), 4)

    str_print_train = 'epoch:' + str(epoch) + ' time:' + str(round(elapsed_time, 1)) + '\t train loss:' + str(
        train_loss)


    if amss_enabled and batch_count > 0:
        avg_params_masked = total_params_masked_epoch / batch_count
        str_print_train += f'\tAMSS avg masked params/batch: {round(avg_params_masked, 0)}'


    PATH_model = path_save_model_base + '/epoch' + str(epoch) + '.pt'
    torch.save(model.state_dict(), PATH_model)


    model.eval()

    val_loader_loss.dataset.ng_sample()
    val_loss = data_utils.metrics_loss(model, val_loader_loss, batch_size)

    testing_loader_loss.dataset.ng_sample()
    test_loss = data_utils.metrics_loss(model, testing_loader_loss, batch_size)

    print(str_print_train + ' val loss:' + str(val_loss) + ' test loss:' + str(test_loss))

    # 可选：每 50 个 epoch 保存一次检查点
    if epoch % 50 == 0 and epoch > 0:
        checkpoint_path = os.path.join(path_save_model_base, f'checkpoint_epoch{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer_bpr.state_dict(),
            'param_importances': param_importances if amss_enabled else None,
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch}")

print("Training completed!")
print(f"Models saved in: {path_save_model_base}")