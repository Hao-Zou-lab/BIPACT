#!/usr/bin/env python
import torch.utils.data
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import scanpy as sc
from BERMUDA1 import training, testing
from pre_processing import pre_processing, read_cluster_similarity
from evaluate import evaluate_scores
from helper import cal_UMAP, plot_labels, plot_expr, plot_loss, gen_dataset_idx

# Set random seed
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)

# IMPORTANT PARAMETER
similarity_thr = 0.5 #S_thr

# nn parameter
code_dim = 200 # vae ae
batch_size = 50 # batch size for each cluster
num_epochs = 120
base_lr = 1e-3
lr_step =  num_epochs/10 # step decay of learning rates
momentum = 0.9
l2_decay = 5e-5
gamma = 1  # regularization between reconstruction and transfer learning
log_interval = 10
non_similar_weight_factor = 5 
k_neighbors = 50

# CUDA
device_id = 0 # ID of GPU to use
cuda = torch.cuda.is_available()
if cuda:
    torch.cuda.set_device(device_id)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

pre_process_paras = {'take_log': False, 'standardization': True, 'scaling': True}
nn_paras = {'code_dim': code_dim, 'batch_size': batch_size, 'num_epochs': num_epochs,
            'base_lr': base_lr, 'lr_step': lr_step,
            'momentum': momentum, 'l2_decay': l2_decay, 'gamma': gamma,
            'cuda': cuda, 'log_interval': log_interval,
            'non_similar_weight_factor': non_similar_weight_factor}

plt.ioff()

if __name__ == '__main__':
    data_folder = 'data/'
    dataset_file_list = ['sample_2M.csv', 'sample_2Y.csv']
    cluster_similarity_file = data_folder + 'best_hits.csv'
    code_save_file = data_folder + 'code_list.pkl'
    dataset_file_list = [data_folder+f for f in dataset_file_list]
    

    # read data
    dataset_list = pre_processing(dataset_file_list, pre_process_paras)
    cluster_pairs = read_cluster_similarity(cluster_similarity_file, similarity_thr)
    nn_paras['num_inputs'] = len(dataset_list[0]['gene_sym'])

    # Training
    model, classifier, loss_total_list, loss_reconstruct_list, loss_transfer_list = training(dataset_list, cluster_pairs, nn_paras)
    plot_loss(loss_total_list, loss_reconstruct_list, loss_transfer_list, data_folder+'loss.png')
    
    # Extract codes and reconstructed data
    code_list, reconstruct_list = testing(model, dataset_list, nn_paras)
    with open(code_save_file,'wb') as f:
        pickle.dump((code_list, reconstruct_list), f)
    
    # Combine datasets in dataset_list
    pre_process_paras = {'take_log': True, 'standardization': False, 'scaling': False}
    dataset_list = pre_processing(dataset_file_list, pre_process_paras)
    cell_list = []
    data_list = []
    cluster_list = []
    for dataset in dataset_list:
        data_list.append(dataset['gene_exp'])
        cell_list.append(dataset['cell_labels'])
        cluster_list.append(dataset['cluster_labels'])
    cell_labels = np.concatenate(cell_list)
    dataset_labels = gen_dataset_idx(data_list)
    cluster_labels = np.concatenate(cluster_list)
    
    # Load code and reconstructed data
    with open(code_save_file,'rb') as f:
        code_list, reconstruct_list = pickle.load(f)
    code = np.concatenate(code_list, axis=1).transpose()  # Shape: [num_samples, code_dim]
    reconstruct = np.concatenate(reconstruct_list, axis=1).transpose()  # Shape: [num_samples, num_features]
    data = np.concatenate(data_list, axis=1).transpose()  # Shape: [num_samples, num_features]

    # Calculate the difference between original data and reconstructed data
    data_diff = data - reconstruct
    
    print("Data:\n", data[:5, :5])
    print("\nReconstruct:\n", reconstruct[:5, :5])
    print("\nData Difference:\n", data_diff[:5, :5])

    # Calculate UMAP
    umap_code = cal_UMAP(code)
    umap_data = cal_UMAP(data)
    umap_recon = cal_UMAP(reconstruct)
    umap_diff = cal_UMAP(data_diff)
    
    # Plot results
    cell_type_dict = {1: 'Periderm',2: 'Phloem', 3: 'Root epidermis t1', 4: 'Root epidermis t2', 5: 'Unknown_2M',
     6: 'Xylem'}
    dataset_dict = {1: '2Y', 2: '2M'}

    # Plot UMAPs
    plot_labels(umap_code, cell_labels, cell_type_dict, ['UMAP_1', 'UMAP_2'], data_folder+'ae_cell_type.png')
    plot_labels(umap_data, cell_labels, cell_type_dict, ['UMAP_1', 'UMAP_2'], data_folder+'uncorrected_cell_type.png')
    plot_labels(umap_diff, cell_labels, cell_type_dict, ['UMAP_1', 'UMAP_2'], data_folder+'diff_cell_type.png')
    plot_labels(umap_recon, cell_labels, cell_type_dict, ['UMAP_1', 'UMAP_2'], data_folder+'recon_cell_type.png')

    plot_labels(umap_code, dataset_labels, dataset_dict, ['UMAP_1', 'UMAP_2'], data_folder + 'ae_dataset.png')
    plot_labels(umap_data, dataset_labels, dataset_dict, ['UMAP_1', 'UMAP_2'], data_folder + 'uncorrected_dataset.png')
    plot_labels(umap_diff, dataset_labels, dataset_dict, ['UMAP_1', 'UMAP_2'], data_folder + 'diff_dataset.png')
    plot_labels(umap_recon, dataset_labels, dataset_dict, ['UMAP_1', 'UMAP_2'], data_folder + 'recon_dataset.png')

   # Create an AnnData object using scanpy
    adata = sc.AnnData(X=code)
    adata.obsm['X_umap'] = umap_code
    adata.obs['batch'] = dataset_labels
    adata.obs['celltype'] = cell_labels

   # SAVE
    output_path = data_folder + 'ae_result.h5ad'
    adata.write(output_path)

    def print_evaluation_scores(source, div_ent_code, sil_code, cell_labels, dataset_labels, num_datasets, kn = k_neighbors):
        print(f'{source}')
        div_score, ent_score, sil_score, batch_asw, kbet_score, clisi_score = evaluate_scores(
            div_ent_code, sil_code, cell_labels, dataset_labels, num_datasets, 20, 20, 'cosine', kn)
        print('divergence_score: {:.3f}, entropy_score: {:.3f}, silhouette_score: {:.3f}, '
              'batch_silhouette_score: {:.3f}, kBET_score: {:.3f}, cLISI_score: {:.3f}'.format(
                div_score, ent_score, sil_score, batch_asw, kbet_score, clisi_score))

    # Evaluation using proposed metrics
    num_datasets = len(dataset_file_list)
    sources = {
        'ae': (umap_code, code),
        'uncorrected': (umap_data, data),
        'difference': (umap_diff, data_diff),
        'recon': (umap_recon, umap_recon)
    }

    for source, codes in sources.items():
        print_evaluation_scores(source, *codes, cell_labels, dataset_labels, num_datasets, kn = k_neighbors)