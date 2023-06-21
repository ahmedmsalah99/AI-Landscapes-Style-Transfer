import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
import random
from taming.data.utils import custom_collate
from taming.data.unpaired_image import UnpairedImageTest


def prepare_data(images_paths,styles_paths,size=256):
    dataset = UnpairedImageTest(size=size)
    dataset.add_paths(images_paths,styles_paths)
    return dataset
def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))
def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def generate_landscape_art(images_paths,styles_paths,ckpt_path,out_path):
    # Initilizations
    name = 'landscape2art'
    base = ''
    postfix = ''
    seed  = random.randint(1, 100000)
    
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # sys.path.append(os.getcwd())
    
    if name:
        name = "_"+name
    elif base:
        cfg_fname = os.path.split(base[0])[-1]
        cfg_name = os.path.splitext(cfg_fname)[0]
        name = "_"+cfg_name
    else:
        name = ""


    seed_everything(seed)

    # model
    model_configs = {'base_learning_rate': 4.5e-06, 'target': 'taming.models.vqgan_ref.VQModel_Ref', 'params': {'ckpt_path': ckpt_path, 'embed_dim': 256, 'n_embed': 1024, 'ddconfig': {'double_z': False, 'z_channels': 256, 'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128, 'ch_mult': [1, 1, 2, 2, 4], 'num_res_blocks': 2, 'attn_resolutions': [16], 'dropout': 0.0}, 'lossconfig': {'target': 'taming.modules.losses.vqperceptual_ref.VQLPIPS_Ref', 'params': {'disc_ndf': 64, 'disc_num_layers': 0, 'disc_conditional': False, 'disc_in_channels': 256, 'disc_start': 0, 'disc_weight': 1.0, 'disc_factor': 0.8, 'codebook1_weight': 1.0, 'codebook2_weight': 1.0, 'reverse_weight': 1.0, 'style_weight': 10.0, 'G_step': 1}}}}
    model = instantiate_from_config(model_configs)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    # run
    device = torch.device('cuda:0')
    model = model.to(device)
    model.eval()
    dataset = prepare_data(images_paths,styles_paths)
    loader = DataLoader(dataset, batch_size=len(styles_paths),collate_fn=custom_collate)
    c=0
    for batch_idx, batch in enumerate(loader):
        print('Testing', batch_idx, '/', len(loader))
        ### Log images ###
        with torch.no_grad():
            results = model.generate_style(batch)
        for i,k in enumerate(results):
            for image in k:
                image = (image+1.0)/2.0
                image = torch.clip(image, 0, 1)
                image = image.transpose(0,1).transpose(1,2)
                image = image.detach().cpu().numpy()
                image = (image*255).astype(np.uint8)
                filename = str(c)+"_{:06}.png".format(batch_idx)     
                c+=1  
                main_path =  os.path.join(out_path,str(i))
                if (not os.path.exists(main_path)):
                    os.mkdir(main_path)
                path = os.path.join(main_path, filename)
                Image.fromarray(image).save(path)
                
        del results

                    

