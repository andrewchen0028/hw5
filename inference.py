import os 
import sys 
import argparse
import numpy as np
import ruamel.yaml as yaml
import torch
import wandb 
import logging 
from logging import getLogger as get_logger
from tqdm import tqdm 
from PIL import Image
import torch.nn.functional as F

from torchvision.utils  import make_grid
from torchvision import datasets, transforms
from torchvision.transforms.functional import pil_to_tensor

from models import UNet, VAE, ClassEmbedder
from schedulers import DDPMScheduler, DDIMScheduler
from pipelines import DDPMPipeline
from utils import seed_everything, load_checkpoint

from train import parse_args

logger = get_logger(__name__)


def main():
    # parse arguments
    args = parse_args()
    
    # seed everything
    seed_everything(args.seed)
    generator = torch.Generator(device='cuda')
    generator.manual_seed(args.seed)
    
    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    
    # setup model
    logger.info("Creating model")
    # unet
    unet = UNet(input_size=args.unet_in_size, input_ch=args.unet_in_ch, T=args.num_train_timesteps, ch=args.unet_ch, ch_mult=args.unet_ch_mult, attn=args.unet_attn, num_res_blocks=args.unet_num_res_blocks, dropout=args.unet_dropout, conditional=args.use_cfg, c_dim=args.unet_ch)
    # preint number of parameters
    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params / 10 ** 6:.2f}M")
    
    # TODO: ddpm shceduler
    scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps,
                              num_inference_steps=args.num_inference_steps,
                              beta_start=args.beta_start,
                              beta_end=args.beta_end,
                              beta_schedule=args.beta_schedule,
                              variance_type=args.variance_type,
                              prediction_type=args.prediction_type,
                              clip_sample=args.clip_sample,
                              clip_sample_range=args.clip_sample_range) # could change parameters here
    # vae 
    vae = None
    if args.latent_ddpm:        
        vae = VAE()
        vae.init_from_ckpt('pretrained/model.ckpt')
        vae.eval()
    # cfg
    class_embedder = None
    if args.use_cfg:
        # TODO: class embeder
        class_embedder = ClassEmbedder(None)
        
    # send to device
    unet = unet.to(device)
    scheduler = scheduler.to(device)
    if vae:
        vae = vae.to(device)
    if class_embedder:
        class_embedder = class_embedder.to(device)
        
    # scheduler
    if args.use_ddim:
        shceduler_class = DDIMScheduler
    else:
        shceduler_class = DDPMScheduler
    # TOOD: scheduler
    scheduler = shceduler_class(num_train_timesteps=args.num_train_timesteps,
                              num_inference_steps=args.num_inference_steps,
                              beta_start=args.beta_start,
                              beta_end=args.beta_end,
                              beta_schedule=args.beta_schedule,
                              variance_type=args.variance_type,
                              prediction_type=args.prediction_type,
                              clip_sample=args.clip_sample,
                              clip_sample_range=args.clip_sample_range) # could change parameters here

    # load checkpoint
    load_checkpoint(unet, scheduler, vae=vae, class_embedder=class_embedder, checkpoint_path=args.ckpt)
    
    # TODO: pipeline
    pipeline = DDPMPipeline(unet, scheduler, vae, class_embedder)

    
    logger.info("***** Running Infrence *****")
    
    # TODO: we run inference to generation 5000 images
    # TODO: with cfg, we generate 50 images per class 
    all_images = []
    if args.use_cfg:
        # generate 50 images per class
        for i in tqdm(range(args.num_classes)):
            logger.info(f"Generating 50 images for class {i}")
            batch_size = 50
            classes = torch.full((batch_size,), i, dtype=torch.long, device=device)
            gen_images = pipeline(batch_size=batch_size, num_inference_steps=args.num_inference_steps, classes=classes, generator=generator, device=device) #TODO
            all_images.append(gen_images)
    else:
        # generate 5000 images
        print('Generate images using pipeline...')
        for _ in tqdm(range(0, 5000, args.batch_size)): # 5000
            gen_images = pipeline(batch_size=args.batch_size, num_inference_steps=args.num_inference_steps, generator=generator, device=device) #TODO
            gen_images_tensor = [pil_to_tensor(i) for i in gen_images]
            all_images.append(torch.stack(gen_images_tensor))
        all_images = torch.vstack(all_images)
    
    # TODO: load validation images as reference batch
    size = 16 if args.reduce_size else args.image_size
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            # transforms.RandomHorizontalFlip(),
            transforms.PILToTensor(),
            # transforms.ToTensor(),
            # transforms.Lambda(lambda x: x.to(torch.uint8)),
            # transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
        ]
    )
    valid_dataset = datasets.ImageFolder(
        root= args.data_dir + '/imagenet100_128x128/validation' if not args.use_CIFAR else args.data_dir + 'CIFAR-10-images/validation',
        transform=transform) 
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=True,
        sampler=None,
        num_workers=args.num_workers,
        persistent_workers=True
    )

    real_images = []
    if args.use_cfg:
        real_images = None
    else:
        for step, (images, labels) in tqdm(enumerate(valid_loader)):
            real_images.append(images)
        real_images = torch.vstack(real_images)
        # print(pipeline.numpy_to_pil(images))

    # TEST
    # real_images = real_images[:32]
    
    # TODO: using torchme   trics for evaluation, check the documents of torchmetrics
    import torchmetrics 
    
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    
    # TODO: compute FID and IS
    fid = FrechetInceptionDistance(feature=64, normalize=False) # normalize=False: value range [0, 255] dtype=uint8

    print('Updating real images...')
    print('real images shape:', real_images.size())
    print('real_images[0]', real_images[0])
    fid.update(real_images, real=True)

    print('Updating generated images...')
    print('generated images shape:', all_images.size())
    print('all_images[0]', all_images[0])
    fid.update(all_images, real=False)

    print(f"FID: {float(fid.compute())}")

    inception_score = InceptionScore(normalize=False)
    inception_score.update(all_images)
    print(f"IS: {inception_score.compute()}")



    
        
    


if __name__ == '__main__':
    main()