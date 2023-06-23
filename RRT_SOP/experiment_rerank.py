import os
import os.path as osp
from copy import deepcopy
from functools import partial
from pprint import pprint

import sacred
import torch
import torch.nn as nn
from sacred import SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
from torch.backends import cudnn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

#from models.ingredient import model_ingredient, get_model
from models_2.ingredient import model_ingredient, get_model
from utils import pickle_load, pickle_save
from utils import state_dict_to_cpu, BinaryCrossEntropyWithLogits, num_of_trainable_params
from utils.data.dataset_ingredient import data_ingredient, get_loaders
from utils.training import train_rerank, evaluate_rerank, train_rerank_backbone, train_rerank_transformer

ex = sacred.Experiment('Rerank (train)', ingredients=[data_ingredient, model_ingredient])
# Filter backspaces and linefeeds
SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def config():
    epochs = 3
    lr = 0.0001
    weight_decay = 4e-4
    scheduler_tau = [60, 80]
    scheduler_gamma = 0.1
    cpu = False  # Force training on CPU
    cudnn_flag = 'benchmark'
    temp_dir = osp.join('logs', 'temp')
    no_bias_decay = False
    resume = None
    cache_nn_inds = '/content/Project_With_ReRanking/RRT_SOP/rrt_sop_caches/rrt_r50_sop_nn_inds_test.pkl'
    seed = 459858808


@ex.capture
def get_optimizer_scheduler(parameters, lr, weight_decay, scheduler_tau, scheduler_gamma):
    optimizer = AdamW(parameters, lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_tau, gamma=scheduler_gamma)
    return optimizer, scheduler


@ex.capture
def get_loss():
    return BinaryCrossEntropyWithLogits()

#################################################################################################################################################

def backbone_train(epochs, cpu, cudnn_flag, temp_dir, seed, no_bias_decay, resume, cache_nn_inds):
    print(f"Training backbone only")
    device = torch.device('cuda:0' if torch.cuda.is_available() and not cpu else 'cpu')
    print(f"Device: {device}")
    if cudnn_flag == 'deterministic':
        setattr(cudnn, cudnn_flag, True)

    torch.manual_seed(seed)
    loaders, recall_ks = get_loaders()

    torch.manual_seed(seed)
    model, transformer = get_model(num_classes=loaders.num_classes)

    if resume is not None:
        state_dict = torch.load(resume, map_location=torch.device('cpu'))
        if 'state' in state_dict:
            state_dict = state_dict['state']
        model.load_state_dict(state_dict, strict=True)
    print('# of trainable parameters: ', num_of_trainable_params(model))
    print('# of trainable parameters of the transformer: ', num_of_trainable_params(transformer))
    class_loss = get_loss()

    # Rerank the top-15 only during training to save time
    cache_nn_inds = pickle_load(cache_nn_inds)[:, :20]
    cache_nn_inds = torch.from_numpy(cache_nn_inds)

    model.to(device)
    transformer.to(device)
    # if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    parameters = []
    if no_bias_decay:
        parameters.append({'params': [par for par in model.parameters() if par.dim() != 1]})
        parameters.append({'params': [par for par in model.parameters() if par.dim() == 1], 'weight_decay': 0})
    else:
        parameters.append({'params': model.parameters()})
    optimizer, scheduler = get_optimizer_scheduler(parameters=parameters)

    # setup partial function to simplify call
    eval_function = partial(evaluate_rerank, backbone=model, transformer=transformer, cache_nn_inds=cache_nn_inds,
        recall_ks=recall_ks, query_loader=loaders.query, gallery_loader=loaders.gallery)

    # setup best validation logger
    metrics = eval_function()[0]
    pprint(metrics)
    best_val = (0, metrics, deepcopy(model.state_dict()))

    torch.manual_seed(seed)

    save_name = "/content/drive/MyDrive/models/backbone.pth"

    for epoch in range(epochs):
        if cudnn_flag == 'benchmark':
            setattr(cudnn, cudnn_flag, True)

        train_rerank_backbone(model=model, loader=loaders.train, optimizer=optimizer, scheduler=scheduler, epoch=epoch, ex=ex)

        # validation
        if cudnn_flag == 'benchmark':
            setattr(cudnn, cudnn_flag, False)
        metrics = eval_function()[0]
        print('Validation [{:03d}]'.format(epoch)), pprint(metrics)
        ex.log_scalar('val.recall@1', metrics[1], step=epoch + 1)

        # save model dict if the chosen validation metric is better
        if metrics[1] >= best_val[1][1]:
            best_val = (epoch + 1, metrics, deepcopy(model.state_dict()))
            torch.save(
                {
                    'state': state_dict_to_cpu(best_val[2]),
                    'optim': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, save_name)
            resume = save_name
    
    print(f"Finished training the backbone")

#################################################################################################################################################

def generate_features(model: nn.Module,
        loader: DataLoader,
        ) -> None:
    
    model.eval()
    device = next(model.parameters()).device
    print(device)
    to_device = lambda x: x.to(device, non_blocking=True)

    features = []

    save_size = 10
    save_order = 0
    arrived_at = -1
    length = 0


    pbar = tqdm(loader, ncols=80, desc='Extracting features...')
    for i, (batch, labels, indices) in enumerate(pbar):
        batch, labels, indices = map(to_device, (batch, labels, indices))

        ##################################################
        ## extract features
        if save_order <= arrived_at:
          length += 1
          if length >= save_size:
            save_order += 1
            length = 0
          continue
        else:
          l = model(batch)
          features.append(l)
          length += 1
        
        if length >= save_size:
            length = 0
            features_to_save = torch.cat(features, 0)
            #print(f"features_to_save dimension: {features_to_save.size()}")
            #print(f"Tensor to save size: {features_to_save.nelement() * features_to_save.element_size() / 1048576} MB")
            torch.save(features_to_save, f"/content/drive/MyDrive/features/features_{save_order}.pt")
            save_order += 1
            #print(f"Free GPU memory before deleting: {get_gpu_memory()}")
            del features
            del features_to_save
            torch.cuda.empty_cache()
            features = []
            #print(f"Free GPU memory after deleting: {get_gpu_memory()}")

    if length < save_size:
        length = 0
        features_to_save = torch.cat(features, 0)
        #print(f"features_to_save dimension: {features_to_save.size()}")
        #print(f"Tensor to save size: {features_to_save.nelement() * features_to_save.element_size() / 1048576} MB")
        torch.save(features_to_save, f"/content/drive/MyDrive/features/features_{save_order}.pt")
        save_order += 1
        #print(f"Free GPU memory before deleting: {get_gpu_memory()}")
        del features
        del features_to_save
        torch.cuda.empty_cache()
        features = []
        #print(f"Free GPU memory after deleting: {get_gpu_memory()}")

    return features


def transformer_train(epochs, cpu, cudnn_flag, temp_dir, seed, no_bias_decay, resume, cache_nn_inds):
    print(f"Training transformer only")
    device = torch.device('cuda:0' if torch.cuda.is_available() and not cpu else 'cpu')
    print(f"Device: {device}")
    if cudnn_flag == 'deterministic':
        setattr(cudnn, cudnn_flag, True)

    torch.manual_seed(seed)
    loaders, recall_ks = get_loaders()

    torch.manual_seed(seed)
    model, transformer = get_model(num_classes=loaders.num_classes)

    resume = "/content/drive/MyDrive/models/backbone.pth"

    if resume is not None:
        state_dict = torch.load(resume, map_location=torch.device('cpu'))
        if 'state' in state_dict:
            state_dict = state_dict['state']
        model.load_state_dict(state_dict, strict=True)
    print('# of trainable parameters: ', num_of_trainable_params(model))
    print('# of trainable parameters of the transformer: ', num_of_trainable_params(transformer))
    class_loss = get_loss()
    
    #backbone = torch.load("/content/drive/MyDrive/models/backbone.pth")

    # Rerank the top-15 only during training to save time
    cache_nn_inds = pickle_load(cache_nn_inds)[:, :20]
    cache_nn_inds = torch.from_numpy(cache_nn_inds)

    model.to(device)
    transformer.to(device)
    
    # if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
        
    parameters = []
    if no_bias_decay:
        parameters.append({'params': [par for par in model.parameters() if par.dim() != 1]})
        parameters.append({'params': [par for par in model.parameters() if par.dim() == 1], 'weight_decay': 0})
    else:
        parameters.append({'params': model.parameters()})
    optimizer, scheduler = get_optimizer_scheduler(parameters=parameters ,lr = 0.001)
    
    with torch.no_grad():
        generate_features(model, loaders.train)

    # setup partial function to simplify call
    eval_function = partial(evaluate_rerank, backbone=model, transformer=transformer, cache_nn_inds=cache_nn_inds,
        recall_ks=recall_ks, query_loader=loaders.query, gallery_loader=loaders.gallery)

    # setup best validation logger
    metrics = eval_function()[0]
    pprint(metrics)
    best_val = (0, metrics, deepcopy(model.state_dict()))

    torch.manual_seed(seed)
    # saving
    save_name = osp.join(temp_dir, 
            '{}_{}.pt'.format(
                        ex.current_run.config['model']['arch'],
                        ex.current_run.config['dataset']['name']
                    )
            )
    os.makedirs(temp_dir, exist_ok=True)
    save_name = "/content/drive/MyDrive/models/transformer.pth"

    for epoch in range(epochs):
        if cudnn_flag == 'benchmark':
            setattr(cudnn, cudnn_flag, True)

        train_rerank_transformer(model=transformer, loader=loaders.train, class_loss=class_loss, optimizer=optimizer, scheduler=scheduler, epoch=epoch, ex=ex)

        # validation
        if cudnn_flag == 'benchmark':
            setattr(cudnn, cudnn_flag, False)
        metrics = eval_function()[0]
        print('Validation [{:03d}]'.format(epoch)), pprint(metrics)
        ex.log_scalar('val.recall@1', metrics[1], step=epoch + 1)

        # save model dict if the chosen validation metric is better
        if metrics[1] >= best_val[1][1]:
            best_val = (epoch + 1, metrics, deepcopy(model.state_dict()))
            torch.save(
                {
                    'state': state_dict_to_cpu(best_val[2]),
                    'optim': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, save_name)

    # logging
    ex.info['recall'] = best_val[1]
    ex.add_artifact(save_name)

    return best_val[1][1]

@ex.automain
def main(epochs, cpu, cudnn_flag, temp_dir, seed, no_bias_decay, resume, cache_nn_inds):
    #backbone_train(epochs, cpu, cudnn_flag, temp_dir, seed, no_bias_decay, resume, cache_nn_inds)
    transformer_train(epochs, cpu, cudnn_flag, temp_dir, seed, no_bias_decay, resume, cache_nn_inds)