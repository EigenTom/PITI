import gradio as gr
import argparse
from einops import rearrange
from pretrained_diffusion import dist_util, logger
from torchvision.utils import make_grid
from pretrained_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from pretrained_diffusion.image_datasets_sketch import get_tensor
from pretrained_diffusion.train_util import TrainLoop
from pretrained_diffusion.glide_util import sample 
import torch
import os
import torch as th
import torchvision.utils as tvu
import torch.distributed as dist
from PIL import Image
import cv2
import numpy as np


def create_argparser():
    defaults = dict(
        data_dir="",
        val_data_dir="",
        model_path="./base_edge.pt",
        sr_model_path="./upsample_edge.pt",
        encoder_path="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=2,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=20000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        sample_c=1.,
        sample_respacing="100",
        uncond_p=0.2,
        num_samples=3,
        finetune_decoder = False,
        mode = '',
        )

    defaults_up = defaults
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    defaults_up.update(model_and_diffusion_defaults(True))
    parser_up = argparse.ArgumentParser()
    add_dict_to_argparser(parser_up, defaults_up)

    return parser, parser_up


def run(image, sammple_c=1.3, num_samples=3, sample_step=100):
    parser, parser_up = create_argparser()
    args = parser.parse_args()
    args_up = parser_up.parse_args()
    dist_util.setup_dist()
    
    # will only use "mask" mode
    args.mode = 'coco'
    args_up.mode = 'coco'
    args.model_path = '/home/axton/axton-workspace/csc2125/model_weights/piti_weights/base_mask.pt'
    args.sr_model_path = '/home/axton/axton-workspace/csc2125/model_weights/piti_weights/upsample_mask.pt'

    args.val_data_dir = image
    args.sample_c = sample_c
    args.num_samples = num_samples
    
    options=args_to_dict(args, model_and_diffusion_defaults(0.).keys())
    model, diffusion = create_model_and_diffusion(**options)
 
    options_up=args_to_dict(args_up, model_and_diffusion_defaults(True).keys())
    model_up, diffusion_up = create_model_and_diffusion(**options_up)
 
    
    print(f'[INF] Loading Model from {args.model_path}')
    model_ckpt = dist_util.load_state_dict(args.model_path, map_location="cpu")
    model.load_state_dict(model_ckpt, strict=True)

    print(f'[INF] Loading super-resolution Model from {args.sr_model_path}')
    model_ckpt2 = dist_util.load_state_dict(args.sr_model_path, map_location="cpu")
    model_up.load_state_dict(model_ckpt2, strict=True)

    model.to(dist_util.dev())
    model_up.to(dist_util.dev())
    model.eval()
    model_up.eval()
    
    pil_image = Image.open(image)
    label_pil = pil_image.convert("RGB").resize((256, 256), Image.NEAREST)
    label_tensor =  get_tensor()(label_pil)
    data_dict = {"ref":label_tensor.unsqueeze(0).repeat(args.num_samples, 1, 1, 1)}
 
    print(f"[INF] Performing Sampling...")
    
    sampled_imgs = []
    grid_imgs = []
    img_id = 0
    while (True):
        if img_id >= args.num_samples:
            break
 
        model_kwargs = data_dict
        with th.no_grad():
            samples_lr =sample(
                glide_model= model,
                glide_options= options,
                side_x= 64,
                side_y= 64,
                prompt=model_kwargs,
                batch_size= args.num_samples,
                guidance_scale=args.sample_c,
                device=dist_util.dev(),
                prediction_respacing= str(sample_step),
                upsample_enabled= False,
                upsample_temp=0.997,
                mode = args.mode,
            )

            samples_lr = samples_lr.clamp(-1, 1)

            tmp = (127.5*(samples_lr + 1.0)).int() 
            model_kwargs['low_res'] = tmp/127.5 - 1.

            samples_hr =sample(
                glide_model= model_up,
                glide_options= options_up,
                side_x=256,
                side_y=256,
                prompt=model_kwargs,
                batch_size=args.num_samples,
                guidance_scale=1,
                device=dist_util.dev(),
                prediction_respacing= "fast27",
                upsample_enabled=True,
                upsample_temp=0.997,
                mode = args.mode,
            )
 
            samples_hr = samples_hr 
      
            for hr in samples_hr:
 
                hr = 255. * rearrange((hr.cpu().numpy()+1.0)*0.5, 'c h w -> h w c')
                sample_img = Image.fromarray(hr.astype(np.uint8))
                sampled_imgs.append(sample_img)
                img_id += 1   

            grid_imgs.append(samples_hr)
            
    # Stack the image tensors into a list
    grid = torch.stack(grid_imgs, 0)
    # rearrange the dimensions of the tensor
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    # convert the tensors from [-1, 1] to [0, 1], then multiply them by 255 to further scale them to [0, 255]
    images = [(255. * rearrange((img+1.0)*0.5, 'c h w -> h w c').cpu().numpy()) for img in grid]
    # convert the numpy arrays to PIL images
    images = [Image.fromarray(img.astype(np.uint8)) for img in images]
    return images
 
 
if __name__ == '__main__':
    image = "/home/axton/axton-workspace/csc2125/models/openseed/output_tmp/full_mask.png"
    sample_c = 1.4
    sample_step = 100
    num_samples = 4
    res_img = run(image, sample_c, num_samples, sample_step)
    
    for i, img in enumerate(res_img):
        img.save(f"/home/axton/axton-workspace/csc2125/synthesized_imgs/sample_{i}.png")
    