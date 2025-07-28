import numpy as np
import torch
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import ae as models
from mmd import mix_rbf_mmd2
import math
import time
import sys
from tqdm import tqdm  
import torch.nn as nn
from Fun import *
# sigma for MMD
base = 1.0
sigma_list = [1, 2, 4, 8, 16]
sigma_list = [sigma / base for sigma in sigma_list]


from imblearn.over_sampling import RandomOverSampler
imblearn_seed = 0



def training(dataset_list, cluster_pairs, nn_paras):
    """ Training an autoencoder to remove batch effects
    Args:
        dataset_list: list of datasets for batch correction
        cluster_pairs: pairs of similar clusters with weights
        nn_paras: parameters for neural network training
    Returns:
        model: trained autoencoder
        loss_total_list: list of total loss
        loss_reconstruct_list: list of reconstruction loss
        loss_transfer_list: list of transfer loss
    """
    # load nn parameters
    batch_size = nn_paras['batch_size']
    num_epochs = nn_paras['num_epochs']
    num_inputs = nn_paras['num_inputs']
    code_dim = nn_paras['code_dim']
    cuda = nn_paras['cuda']
    

    # training data for autoencoder, construct a DataLoader for each cluster
    cluster_loader_dict = {}
    for i in range(len(dataset_list)):
        gene_exp = dataset_list[i]['gene_exp'].transpose()
        cluster_labels = dataset_list[i]['cluster_labels']
        cell_labels = dataset_list[i]['cell_labels']  # cluster labels do not overlap between datasets
        study_label = dataset_list[i]['study_label'] 
        unique_labels = np.unique(cluster_labels)
        # Random oversampling based on cell cluster sizes

        unique_labels_before = np.unique(cluster_labels, return_counts=True)
        print(f"Before Oversampling - Cluster counts: {dict(zip(unique_labels_before[0], unique_labels_before[1]))}")


        ros = RandomOverSampler(random_state=imblearn_seed)
        gene_exp, cluster_labels = ros.fit_resample(gene_exp, cluster_labels)
        resampled_indices = ros.sample_indices_
     
        unique_labels_after = np.unique(cluster_labels, return_counts=True)
        print(f"After Oversampling - Cluster counts: {dict(zip(unique_labels_after[0], unique_labels_after[1]))}")

        # 使用重采样的索引扩展 sample_labels
        study_label = study_label[resampled_indices]
        cell_labels = cell_labels[resampled_indices]  # 新增这一步以处理 cell_labels

        unique_labels = np.unique(cluster_labels)
        for j in range(len(unique_labels)):
            idx = cluster_labels == unique_labels[j]
            if cuda:
                torch_dataset = torch.utils.data.TensorDataset(
                    torch.FloatTensor(gene_exp[idx, :]).cuda(),
                    torch.LongTensor(cluster_labels[idx]).cuda(),
                    torch.FloatTensor(study_label[idx, :]).cuda(),
                    torch.LongTensor(cell_labels[idx]).cuda())
            else:
                torch_dataset = torch.utils.data.TensorDataset(
                    torch.FloatTensor(gene_exp[idx, :]),
                    torch.LongTensor(cluster_labels[idx]),
                    torch.FloatTensor(study_label[idx, :]),
                    torch.LongTensor(cell_labels[idx]))

            data_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=batch_size,
                                              shuffle=True, drop_last=True)
            cluster_loader_dict[unique_labels[j]] = data_loader

    code_size = gene_exp.shape[1]
    print("Code size based on the number of features:", code_size)
    # create model
    if code_dim == 20:
        model = models.autoencoder_20(num_inputs=num_inputs)
    elif code_dim == 2:
        model = models.autoencoder_2(num_inputs=num_inputs)
    elif code_dim == 200:
        model = models.VAE(num_inputs=num_inputs)
    elif code_dim == 50:
        model = models.VAE50(num_inputs=num_inputs)
    else:
        model = models.autoencoder_20(num_inputs=num_inputs)
    if cuda:
        model.cuda()
        print('cuda··Yes!!')

    classifier = Classifier(input_dim=code_size, output_dim=2)

    if cuda:
        model.cuda()
        classifier = classifier.cuda()




    # training
    print("Starting training")
    loss_total_list = []  # list of total loss
    loss_reconstruct_list = []
    loss_transfer_list = []
    loss_kl_list = []
    loss_cos_list = []
    # 初始化整个训练过程的进度条
    pbar = tqdm(total=num_epochs, dynamic_ncols=True, ascii=True, desc='Training')

    for epoch in range(1, num_epochs + 1):
        avg_loss, avg_reco_loss, avg_tran_loss, avg_kl_loss, avg_cosine_sim_loss = training_epoch(
            epoch, model, cluster_loader_dict, cluster_pairs, nn_paras, classifier)
        # terminate early if loss is nan
        if math.isnan(avg_reco_loss) or math.isnan(avg_tran_loss):
            pbar.close()
            return [], model, [], [], []
        loss_total_list.append(avg_loss)
        loss_reconstruct_list.append(avg_reco_loss)
        loss_transfer_list.append(avg_tran_loss)
        loss_kl_list.append(avg_kl_loss)
        loss_cos_list.append(avg_cosine_sim_loss)

        # 更新进度条的描述和损失值
        pbar.set_description(f"Epoch {epoch}/{num_epochs}")
        pbar.set_postfix({
            'loss': f'{avg_loss:.2f}',
            'reco_loss': f'{avg_reco_loss:.2f}',
            'tran_loss': f'{avg_tran_loss:.2f}',
            'kl_loss': f'{avg_kl_loss:.2f}',
            'cosine_sim_loss': f'{avg_cosine_sim_loss:.2f}',
        })
        pbar.update(1)  # 进度条前进一格

    pbar.close()

    return model,classifier, loss_total_list, loss_reconstruct_list, loss_transfer_list

def training_epoch(epoch, model, cluster_loader_dict, cluster_pairs, nn_paras, classifier):
    """ Training an epoch
        Args:
            epoch: number of the current epoch
            model: autoencoder
            cluster_loader_dict: dict of DataLoaders indexed by clusters
            cluster_pairs: pairs of similar clusters with weights
            nn_paras: parameters for neural network training
        Returns:
            avg_total_loss: average total loss of mini-batches
            avg_reco_loss: average reconstruction loss of mini-batches
            avg_tran_loss: average transfer loss of mini-batches
            avg_kl_loss: average KL divergence loss of mini-batches
    """
    log_interval = nn_paras['log_interval']
    base_lr = nn_paras['base_lr']
    lr_step = nn_paras['lr_step']
    num_epochs = nn_paras['num_epochs']
    l2_decay = nn_paras['l2_decay']
    gamma = nn_paras['gamma']
    cuda = nn_paras['cuda']
    non_similar_weight_factor = nn_paras['non_similar_weight_factor']

    criterion = torch.nn.CrossEntropyLoss()

    learning_rate = base_lr / math.pow(2, math.floor(epoch / lr_step))
    gamma_rate = 2 / (1 + math.exp(-10 * (epoch) / num_epochs)) - 1
    
    gamma1 = gamma_rate * gamma
    gamma_rate2 = 0.00001 * (2 / (1 + math.exp(-3 * epoch / num_epochs)) - 1)
    gamma2 = gamma_rate2 * gamma
    gamma_rate3 = 10 * math.exp(-9 * epoch / num_epochs) + 1
    gamma3 = gamma_rate3 * gamma

    # 不再在此函数内打印信息
    optimizer = torch.optim.Adam([
        {'params': model.encoder.parameters()},
        {'params': model.decoder.parameters()},
    ], lr=learning_rate, weight_decay=l2_decay)
    #optimizer_classifier = torch.optim.Adam(classifier.parameters(), lr=learning_rate) 

    model.train()

    iter_data_dict = {cls: iter(cluster_loader_dict[cls]) for cls in cluster_loader_dict}
    num_iter = max(len(cluster_loader_dict[cls]) for cls in cluster_loader_dict)

    total_loss = 0
    total_reco_loss = 0
    total_tran_loss = 0
    total_kl_loss = 0
    num_batches = 0
    #total_pred_loss = 0
    total_cosine_sim_loss = 0

    
    for it in range(num_iter):
        data_dict = {}
        label_dict = {}
        study_dict = {}
        code_dict = {}
        reconstruct_dict = {}
        mu_dict = {}
        logvar_dict = {}
        #pred_dict = {}
        cell_dict = {}
        global_codes = []
        global_cell_types = []
        global_studies = []

        for cls in cluster_loader_dict:
            # Check if the iterator has been exhausted
            if cls not in iter_data_dict or iter_data_dict[cls] is None:
                iter_data_dict[cls] = iter(cluster_loader_dict[cls])

            try:
                data, labels, study, cell = iter_data_dict[cls].__next__()
            except StopIteration:
                # Reinitialize the iterator if exhausted
                iter_data_dict[cls] = iter(cluster_loader_dict[cls])
                data, labels, study, cell = iter_data_dict[cls].__next__()

            data_dict[cls] = Variable(data)
            label_dict[cls] = Variable(labels)
            study_dict[cls] = Variable(study)
            cell_dict[cls] = Variable(cell)

        for cls in data_dict:
            noisy_input = add_noise(data_dict[cls], p=0.2)
            code, mu, logvar, reconstruct = model(noisy_input)
            #pred = classifier(reconstruct)
            code_dict[cls] = code
            reconstruct_dict[cls] = reconstruct
            mu_dict[cls] = mu
            logvar_dict[cls] = logvar
            #pred_dict[cls] = pred
            global_codes.append(code)
            global_cell_types.append(cell_dict[cls])
            global_studies.append(study_dict[cls])

        optimizer.zero_grad()
        #optimizer_classifier.zero_grad()

        # Transfer loss calculation
        loss_transfer = torch.FloatTensor([0])
        if cuda:
            loss_transfer = loss_transfer.cuda()
        for i in range(cluster_pairs.shape[0]):
            cls_1 = int(cluster_pairs[i, 0])
            cls_2 = int(cluster_pairs[i, 1])
            if cls_1 not in code_dict or cls_2 not in code_dict:
                continue
            mmd2_D = mix_rbf_mmd2(code_dict[cls_1], code_dict[cls_2], sigma_list)
            loss_transfer += mmd2_D * cluster_pairs[i, 2]

        # Reconstruction loss calculation
        loss_reconstruct = torch.FloatTensor([0])
        if cuda:
            loss_reconstruct = loss_reconstruct.cuda()
        for cls in data_dict:
            loss_reconstruct += F.mse_loss(reconstruct_dict[cls], data_dict[cls])
        
        # loss_cosine_sim  loss calculation
        #loss_cosine_sim  = torch.FloatTensor([0])
        #if cuda:
        #    loss_cosine_sim  = loss_cosine_sim .cuda()
        #for cls in data_dict:
        #    loss_cosine_sim  += cosine_similarity_loss(code_dict[cls], cell_dict[cls])
        #    print(cell_dict[cls])
            # 合并所有编码向量和细胞类型

        global_codes = torch.cat(global_codes, dim=0)
        global_cell_types = torch.cat(global_cell_types, dim=0)
        global_studies = torch.cat(global_studies, dim=0)

        unique_cell_types = global_cell_types.unique().to(global_cell_types.device)
        common_cell_types = []

        for cell_type in unique_cell_types:
            cell_type = cell_type.to(global_cell_types.device)  # 将 cell_type 移动到相同设备
            study_values = global_studies[global_cell_types == cell_type]
    
            study_values = study_values.to(global_studies.device)

            if study_values.unique().size(0) == global_studies.unique().size(0):
                common_cell_types.append(cell_type)

        mask = torch.zeros(global_cell_types.size(), dtype=torch.bool, device=global_cell_types.device)
        for cell_type in common_cell_types:
            cell_type = cell_type.to(global_cell_types.device)  # 将 cell_type 移动到相同设备
            mask |= (global_cell_types == cell_type)

        filtered_codes = global_codes[mask]
        filtered_cell_types = global_cell_types[mask]

        loss_cosine_sim = cosine_similarity_loss(filtered_codes, filtered_cell_types, non_similar_weight_factor)

        
        # Pred loss calculation
        #loss_Pred = torch.FloatTensor([0])
        
        #if cuda:
        #    loss_Pred = loss_Pred.cuda()
        #for cls in data_dict:
        #    loss_Pred += F.mse_loss(pred_dict[cls], study_dict[cls])            
            
        # 计算 KL 散度损失
        kl_loss = torch.FloatTensor([0])
        if cuda:
            kl_loss = kl_loss.cuda()
        for cls in mu_dict:
            kl_loss += -0.5 * torch.sum(1 + logvar_dict[cls] - mu_dict[cls].pow(2) - logvar_dict[cls].exp())

        # Total loss includes KL loss
        #loss = 10.1 * loss_reconstruct + 2 * gamma * loss_Pred + kl_loss * gamma2 + 0.05 * loss_cosine_sim
        #loss = gamma3 * 2.0 * loss_reconstruct +  0.1 * gamma1 * loss_transfer + kl_loss* 1 * gamma2 + num_batches * 0.00001 * loss_cosine_sim
        #loss = gamma3 * 3.0 * loss_reconstruct +  0.2 * gamma1 * loss_transfer + kl_loss* 1 * gamma2 + num_batches * 0.001 * loss_cosine_sim
        loss = gamma3 * 3.0 * loss_reconstruct +  0.2 * gamma1 * loss_transfer + kl_loss* 1 * gamma2 + num_batches * 0.02 * loss_cosine_sim

        loss.backward()
        optimizer.step()
        #optimizer_classifier.step()

        # Update total loss
        num_batches += 1
        total_loss += loss.item()
        total_reco_loss += loss_reconstruct.item()
        total_tran_loss += loss_transfer.item()
        total_kl_loss += kl_loss.item()
        #total_pred_loss += loss_Pred.item()
        total_cosine_sim_loss += loss_cosine_sim.item()        
        # 不在此处更新进度条

    avg_total_loss = total_loss / num_batches
    avg_reco_loss = total_reco_loss / num_batches
    avg_tran_loss = total_tran_loss / num_batches
    avg_kl_loss = total_kl_loss / num_batches
    avg_cosine_sim_loss = total_cosine_sim_loss / num_batches
    #avg_tran_loss = total_pred_loss / num_batches
    # 不在此处打印信息

    return avg_total_loss, avg_reco_loss, avg_tran_loss, avg_kl_loss , avg_cosine_sim_loss

def testing(model, dataset_list, nn_paras):
    """ Testing the model to extract codes and reconstructed data
    Args:
        model: autoencoder
        dataset_list: list of datasets for batch correction
        nn_paras: parameters for neural network training
    Returns:
        code_list: list of embedded codes
        reconstruct_list: list of reconstructed data
    """

    # Load neural network parameters
    cuda = nn_paras['cuda']

    data_loader_list = []
    num_cells = []
    for dataset in dataset_list:
        torch_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(dataset['gene_exp'].transpose()), torch.LongTensor(dataset['cell_labels']))
        data_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=len(dataset['cell_labels']),
                                                  shuffle=False)
        data_loader_list.append(data_loader)
        num_cells.append(len(dataset["cell_labels"]))
    model.eval()

    code_list = []  # List of embedded codes
    reconstruct_list = []  # List of reconstructed data
    for i in range(len(data_loader_list)):
        idx = 0
        with torch.no_grad():
            for data, labels in data_loader_list[i]:
                if cuda:
                    data, labels = data.cuda(), labels.cuda()
                # Get code and reconstructed data from the model
                code_tmp, _, _, reconstruct_tmp = model(data)
                code_tmp = code_tmp.cpu().numpy()
                reconstruct_tmp = reconstruct_tmp.cpu().numpy()
                if idx == 0:
                    code = np.zeros((code_tmp.shape[1], num_cells[i]))
                    reconstruct = np.zeros((reconstruct_tmp.shape[1], num_cells[i]))
                code[:, idx:idx + code_tmp.shape[0]] = code_tmp.T
                reconstruct[:, idx:idx + reconstruct_tmp.shape[0]] = reconstruct_tmp.T
                idx += code_tmp.shape[0]
        code_list.append(code)
        reconstruct_list.append(reconstruct)

    return code_list, reconstruct_list

