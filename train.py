import os
import sys
import argparse
import numpy as np
import gc
import ruamel.yaml as yaml
import torch
import wandb
import logging
from logging import getLogger as get_logger
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F

from torchvision import datasets, transforms
from torchvision.utils import make_grid

from models import UNet, VAE, ClassEmbedder
from schedulers import DDPMScheduler, DDIMScheduler
from pipelines import DDPMPipeline
from utils import seed_everything, init_distributed_device, is_primary, AverageMeter, str2bool, save_checkpoint


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model.")

    # config file
    parser.add_argument("--config", type=str, default='hw5_student_starter_code_win/configs/ddpm.yaml',
                        help="config file used to specify parameters")

    # data
    parser.add_argument("--data_dir", type=str,
                        default='Data/', help="data folder")
    parser.add_argument("--image_size", type=int,
                        default=128, help="image size")
    parser.add_argument("--batch_size", type=int,
                        default=4, help="per gpu batch size")
    parser.add_argument("--num_workers", type=int,
                        default=8, help="batch size")
    parser.add_argument("--num_classes", type=int, default=100,
                        help="number of classes in dataset")
    parser.add_argument("--use_CIFAR", type=str2bool,default=False,
                        help="use CIFAR-10 dataset for training")
    parser.add_argument("--reduce_size", type=str2bool,default=False,
                        help="reduce image pixels to speed up training")

    # training
    parser.add_argument("--run_name", type=str, default=None, help="run_name")
    parser.add_argument("--output_dir", type=str,
                        default="hw5_student_starter_code_win/experiments", help="output folder")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float,
                        default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float,
                        default=1e-4, help="weight decay")
    parser.add_argument("--grad_clip", type=float,
                        default=1.0, help="gradient clip")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--mixed_precision", type=str, default='none',
                        choices=['fp16', 'bf16', 'fp32', 'none'], help='mixed precision')

    # ddpm
    parser.add_argument("--num_train_timesteps", type=int,
                        default=1000, help="ddpm training timesteps")
    parser.add_argument("--num_inference_steps", type=int,
                        default=200, help="ddpm inference timesteps")
    parser.add_argument("--beta_start", type=float,
                        default=0.0002, help="ddpm beta start")
    parser.add_argument("--beta_end", type=float,
                        default=0.02, help="ddpm beta end")
    parser.add_argument("--beta_schedule", type=str,
                        default='linear', help="ddpm beta schedule")
    parser.add_argument("--variance_type", type=str,
                        default='fixed_small', help="ddpm variance type")
    parser.add_argument("--prediction_type", type=str,
                        default='epsilon', help="ddpm epsilon type")
    parser.add_argument("--clip_sample", type=str2bool, default=True,
                        help="whether to clip sample at each step of reverse process")
    parser.add_argument("--clip_sample_range", type=float,
                        default=1.0, help="clip sample range")

    # unet
    parser.add_argument("--unet_in_size", type=int,
                        default=128, help="unet input image size")
    parser.add_argument("--unet_in_ch", type=int, default=3,
                        help="unet input channel size")
    parser.add_argument("--unet_ch", type=int, default=128,
                        help="unet channel size")
    parser.add_argument("--unet_ch_mult", type=int,
                        default=[1, 2, 2, 2], nargs='+', help="unet channel multiplier")
    parser.add_argument("--unet_attn", type=int,
                        default=[1, 2, 3], nargs='+', help="unet attantion stage index")
    parser.add_argument("--unet_num_res_blocks", type=int,
                        default=2, help="unet number of res blocks")
    parser.add_argument("--unet_dropout", type=float,
                        default=0.0, help="unet dropout")

    # vae
    parser.add_argument("--latent_ddpm", type=str2bool,
                        default=False, help="use vqvae for latent ddpm")

    # cfg
    parser.add_argument("--use_cfg", type=str2bool, default=False,
                        help="use cfg for conditional (latent) ddpm")
    parser.add_argument("--cfg_guidance_scale", type=float,
                        default=1.0, help="cfg for inference")

    # ddim sampler for inference
    parser.add_argument("--use_ddim", type=str2bool,
                        default=False, help="use ddim sampler for inference")

    # checkpoint path for inference
    parser.add_argument("--ckpt", type=str, default=None,
                        help="checkpoint path for inference")

    # gradient accumulation steps
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of steps to accumulate gradients over")

    # first parse of command-line args to check for config file
    args = parser.parse_args()

    # If a config file is specified, load it and set defaults
    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f:
            file_yaml = yaml.YAML()
            config_args = file_yaml.load(f)
            parser.set_defaults(**config_args)

    # re-parse command-line args to overwrite with any command-line inputs
    args = parser.parse_args()
    return args


def main():

    # parse arguments
    args = parse_args()

    # seed everything
    seed_everything(args.seed)

    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # setup distributed initialize and device
    device = init_distributed_device(args)
    if args.distributed:
        logger.info(
            'Training in distributed mode with multiple processes, 1 device per process.'
            f'Process {args.rank}, total {args.world_size}, device {args.device}.')
    else:
        logger.info(
            f'Training with a single process on 1 device ({args.device}).')
    assert args.rank >= 0

    gc.collect()
    torch.cuda.empty_cache()

    # setup dataset
    logger.info("Creating dataset")
    # : use transform to normalize your images to [-1, 1]
    # : you can also use horizontal flip
    size = 16 if args.reduce_size else args.image_size
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
        ]
    )
    # : use image folder for your train dataset

    # Note: data_dir = path to /Data folder
    train_dataset = datasets.ImageFolder(
        root= args.data_dir + '/imagenet100_128x128/train' if not args.use_CIFAR else args.data_dir + 'CIFAR-10-images/train',
        transform=transform)

    # : setup dataloader
    sampler = None
    if args.distributed:
        # : distributed sampler
        sampler = None
    # : shuffle
    shuffle = False if sampler else True
    #  dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=shuffle,
        sampler=sampler,
        persistent_workers=True
    )

    # calculate total batch_size
    total_batch_size = args.batch_size * args.world_size
    args.total_batch_size = total_batch_size

    # setup experiment folder
    if args.run_name is None:
        args.run_name = f'exp-{len(os.listdir(args.output_dir))}'
    else:
        args.run_name = f'exp-{len(os.listdir(args.output_dir))}-{args.run_name}'
    output_dir = os.path.join(args.output_dir, args.run_name)
    save_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(args.output_dir, exist_ok=True)
    if is_primary(args):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

    # setup model
    logger.info("Creating model")
    # unet
    unet = UNet(input_size=args.unet_in_size, input_ch=args.unet_in_ch, T=args.num_train_timesteps, ch=args.unet_ch, ch_mult=args.unet_ch_mult,
                attn=args.unet_attn, num_res_blocks=args.unet_num_res_blocks, dropout=args.unet_dropout, conditional=args.use_cfg, c_dim=args.unet_ch)
    # preint number of parameters
    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params / 10 ** 6:.2f}M")

    # : ddpm shceduler
    scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps,
                              num_inference_steps=args.num_inference_steps,
                              beta_start=args.beta_start,
                              beta_end=args.beta_end,
                              beta_schedule=args.beta_schedule,
                              variance_type=args.variance_type,
                              prediction_type=args.prediction_type,
                              clip_sample=args.clip_sample,
                              clip_sample_range=args.clip_sample_range)

    # NOTE: this is for latent DDPM
    vae = None
    if args.latent_ddpm:
        vae = VAE()
        # NOTE: do not change this
        vae.init_from_ckpt('pretrained/model.ckpt')
        vae.eval()

    # Note: this is for cfg
    class_embedder = None
    if args.use_cfg:
        # TODO:
        class_embedder = ClassEmbedder(None)

    # send to device
    unet = unet.to(device)
    scheduler = scheduler.to(device)
    if vae:
        vae = vae.to(device)
    if class_embedder:
        class_embedder = class_embedder.to(device)

    # : setup optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # : setup scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-6)

    # max train steps
    num_update_steps_per_epoch = len(train_loader)
    args.max_train_steps = args.num_epochs * num_update_steps_per_epoch

    #  setup distributed training
    if args.distributed:
        unet = torch.nn.parallel.DistributedDataParallel(
            unet, device_ids=[args.device], output_device=args.device, find_unused_parameters=False)
        unet_wo_ddp = unet.module
        if class_embedder:
            class_embedder = torch.nn.parallel.DistributedDataParallel(
                class_embedder, device_ids=[args.device], output_device=args.device, find_unused_parameters=False)
            class_embedder_wo_ddp = class_embedder.module
    else:
        unet_wo_ddp = unet
        class_embedder_wo_ddp = class_embedder
    vae_wo_ddp = vae
    # TODO: setup ddim
    if args.use_ddim:
        scheduler_wo_ddp = DDIMScheduler(num_train_timesteps=args.num_train_timesteps,
                                        num_inference_steps=args.num_inference_steps,
                                        beta_start=args.beta_start,
                                        beta_end=args.beta_end,
                                        beta_schedule=args.beta_schedule,
                                        variance_type=args.variance_type,
                                        prediction_type=args.prediction_type,
                                        clip_sample=args.clip_sample,
                                        clip_sample_range=args.clip_sample_range)
    else:
        scheduler_wo_ddp = scheduler

    # : setup evaluation pipeline
    # NOTE: this pipeline is not differentiable and only for evaluatin
    pipeline = DDPMPipeline(unet, scheduler_wo_ddp, vae, class_embedder)

    # dump config file
    if is_primary(args):
        experiment_config = vars(args)
        with open(os.path.join(output_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
            # Use the round_trip_dump method to preserve the order and style
            file_yaml = yaml.YAML()
            file_yaml.dump(experiment_config, f)

    # start tracker
    if is_primary(args):
        wandb.login()
        wandb_logger = wandb.init(
            project='ddpm',
            name=args.run_name,
            config=vars(args))

    # Start training
    if is_primary(args):
        logger.info("***** Training arguments *****")
        logger.info(args)
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Total optimization steps per epoch {num_update_steps_per_epoch}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps),
                        disable=not is_primary(args))

    # training
    for epoch in range(args.num_epochs):

        # set epoch for distributed sampler, this is for distribution training
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        args.epoch = epoch
        if is_primary(args):
            logger.info(f"Epoch {epoch+1}/{args.num_epochs}")

        loss_m = AverageMeter()

        # : set unet and scheduelr to train
        unet.train()
        scheduler.train()

        # : finish this
        for step, (images,labels) in enumerate(train_loader):

            batch_size = images.size(0)

            # : send to device
            images = images.to(device)
            labels = labels.to(device)

            # NOTE: this is for latent DDPM
            if vae is not None:
                # use vae to encode images as latents
                images = None
                # NOTE: do not change  this line, this is to ensure the latent has unit std
                images = images * 0.1845

            # : zero grad optimizer
            optimizer.zero_grad()

            # NOTE: this is for CFG
            if class_embedder is not None:
                # TODO: use class embedder to get class embeddings
                class_emb = None
            else:
                # NOTE: if not cfg, set class_emb to None
                class_emb = None

            # : sample noise
            noise = torch.randn_like(images).chunk(2)[0].repeat(2, 1, 1, 1)

            # : sample timestep t
            timesteps = torch.randint(
                0, scheduler.num_train_timesteps, (batch_size,), device=device, dtype=torch.long
            )

            # : add noise to images using scheduler
            
            
            
            noisy_images = scheduler.add_noise(images, noise, timesteps)
            

            # : model prediction
            model_pred = unet.forward(noisy_images, timesteps)

            if args.prediction_type == 'epsilon':
                target = noise

            # : calculate loss
            loss = F.mse_loss(model_pred.float(),
                              target.float(), reduction="none")
            loss = loss.mean()
            # record loss
            loss_m.update(loss.item())

            # backward and step
            loss.backward()
            # : grad clip
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 0.1)

            # : step your optimizer
            lr_scheduler.step()

            progress_bar.update(1)

            # logger
            if step % 100 == 0 and is_primary(args):
                logger.info(f"Epoch {epoch+1}/{args.num_epochs}, Step {step}/{num_update_steps_per_epoch}, Loss {loss.item()} ({loss_m.avg})")
                wandb_logger.log({'loss': loss_m.avg})

            # Only optimize after accumulating gradients
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Add near the start of training loop
            if step == 0 and epoch == 0:
                logger.info(f"Image range: {images.min():.3f} to {images.max():.3f}")
                logger.info(f"Noise range: {noise.min():.3f} to {noise.max():.3f}")
                logger.info(f"Model pred range: {model_pred.min():.3f} to {model_pred.max():.3f}")

        # validation
        # send unet to evaluation mode
        unet.eval()
        generator = torch.Generator(device=device)
        generator.manual_seed(epoch + args.seed)

        # NOTE: this is for CFG
        if args.use_cfg:
            # random sample 4 classes
            classes = torch.randint(0, args.num_classes, (4,), device=device)
            # TODO: fill pipeline
            gen_images = pipeline(None)
        else:
            # : fill pipeline
            gen_images = pipeline(batch_size=args.batch_size,
                                  num_inference_steps=args.num_inference_steps,
                                  # classes=args.num_classes,
                                  # guidance_scale=args.cfg_guidance_scale,
                                  generator=generator,
                                  device=device)

        # create a blank canvas for the grid
        grid_image = Image.new(
            'RGB', (4 * args.image_size, 1 * args.image_size))
        # paste images into the grid
        for i, image in enumerate(gen_images):
            x = (i % 4) * args.image_size
            y = 0
            grid_image.paste(image, (x, y))

        # Send to wandb
        if is_primary(args):
            wandb_logger.log({'gen_images': wandb.Image(grid_image)})

        # save checkpoint
        if is_primary(args):
            save_checkpoint(unet_wo_ddp, scheduler_wo_ddp, vae_wo_ddp,
                            class_embedder, optimizer, epoch, save_dir=save_dir)

    # Add after dataloader creation
    logger.info(f"Steps per epoch: {len(train_loader)}")
    logger.info(f"Total training steps: {args.max_train_steps}")


if __name__ == '__main__':
    main()
