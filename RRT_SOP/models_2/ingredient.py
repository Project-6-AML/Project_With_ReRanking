from sacred import Ingredient

from .matcher import MatchERT
from .resnet import resnet18

model_ingredient = Ingredient('model')


@model_ingredient.config
def config():
    arch = 'resnet18'
    pretrained = True  # use a pretrained model from torchvision
    dropout = 0.
    norm_layer = None  # use a normalization layer (batchnorm or layernorm) for the features
    remap = False  # remap features through a linear layer
    detach = False  # detach features before feeding to the classification layer. Prevents training of the feature extractor with cross-entropy.
    normalize = False  # normalize the features
    set_bn_eval = True  # set bn in eval mode even in training
    normalize_weight = False  # normalize the weights of the classification layer
    freeze_backbone = False
    ###############################
    num_global_features = 128  # dimensionality of the features produced by the feature extractor
    num_local_features = 128
    ###############################
    ## Encoder from Transformer (ERT)
    ert_seq_len=102
    ert_dim_feedforward=1024
    ert_nhead=4
    ert_num_encoder_layers=3
    ert_dropout=0.0 
    ert_activation="relu"
    ert_normalize_before=False


@model_ingredient.named_config
def resnet50():
    arch = 'resnet50'
    normalize = True

@model_ingredient.capture
def get_model(
        num_classes, arch, pretrained, num_global_features, norm_layer, detach, remap, normalize, normalize_weight, set_bn_eval, dropout, num_local_features, freeze_backbone,
        ert_seq_len, ert_dim_feedforward, ert_nhead, ert_num_encoder_layers, ert_dropout, ert_activation, ert_normalize_before):
    backbone = resnet18(pretrained=pretrained, num_local_features=num_local_features)
    transformer = MatchERT(d_global=num_global_features, d_model=num_local_features, 
            nhead=ert_nhead, num_encoder_layers=ert_num_encoder_layers, 
            dim_feedforward=ert_dim_feedforward, dropout=ert_dropout, 
            activation=ert_activation, normalize_before=ert_normalize_before)
    
    return backbone, transformer