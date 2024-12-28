"""make variations of input image"""

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything
import sys
import torchvision.transforms as transforms
sys.path.append('/data2/renyulin/Prompt_Diffusion')
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import math
import copy
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def resize(image):
    # Resize = transforms.Resize(size=(512, 512))
    Resize= transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224)
    ])
    resized_image = Resize(image)
    return resized_image

def resize_64(image):
    # Resize = transforms.Resize(size=(512, 512))
    Resize= transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64)
    ])
    resized_image = Resize(image)
    return resized_image

def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]   #[250,]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def assign_concatenated_prompts(model, loaded_weights, model_num):
    prompts = ['prompt_jpeg', 'prompt_vvc', 'prompt_hevc', 'prompt_psnr', 'prompt_ssim', 'prompt_hific']
    prompt_tensors = [loaded_weights[f'Prompt_model{model_num}.{p}'] for p in prompts]
    concatenated = torch.cat(prompt_tensors, dim=0)  # Concatenate along the 0th dimension
    structcond_stage = getattr(model, 'structcond_stage_model')
    prompt_model = getattr(structcond_stage, f'Prompt_model{model_num}')
    setattr(prompt_model, 'prompt_param', torch.nn.Parameter(concatenated))
   

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    # model.cuda()
    model.eval()
    return model

def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

dataset = ['JPEG_10','JPEG_15','JPEG_20','VTM_47','VTM_42','VTM_37','HM_47','HM_42','HM_37','WEBP_1','WEBP_5','WEBP_10','PSNR_1','PSNR_2','PSNR_3','SSIM_1','SSIM_2','SSIM_3','HIFIC_1','HIFIC_2','HIFIC_3']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ddpm_steps",
        type=int,
        default=200,
        help="number of ddpm sampling steps",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='Output/LIVE1_test/',
        help="number of ddpm sampling steps",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default='/data1/renyulin/Prompt_Diffusion/LIVE1/',
        help="number of ddpm sampling steps",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=2,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/data1/renyulin/MoE-DiffIR/configs/MoE-DiffIR/MoE-DiffIR_Only_MoE_Prompt.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
          default='N7_K3.ckpt',
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--klvae_ckpt",
        type=str,
        default="Decoder_Finetuned.ckpt",
        help="path to checkpoint of klvae model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=256,
        help="input size",
    )
    opt = parser.parse_args()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    klvae_config = OmegaConf.load("configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml")
    vq_model = load_model_from_config(klvae_config,opt.klvae_ckpt)
    vq_model = vq_model.to(device)
    seed_everything(opt.seed)
    transform = torchvision.transforms.Compose([
        torchvision.transforms. Resize(opt.input_size),
        torchvision.transforms.CenterCrop(opt.input_size),
    ])
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    model = model.to(device)
    for a in range(21):
        out_dir = opt.output_dir+dataset[a]
        os.makedirs(out_dir, exist_ok=True)
        outpath = out_dir
        batch_size = opt.n_samples
        init_img = opt.input_dir + dataset[a]
        img_list_ori = os.listdir(init_img)
        img_list = copy.deepcopy(img_list_ori)
        init_image_list = []
        for item in img_list_ori:
            if os.path.exists(os.path.join(outpath, item)):
                img_list.remove(item)
                continue
            cur_image = load_img(os.path.join(init_img, item)).to(device)
            cur_image = transform(cur_image)
            cur_image = cur_image.clamp(-1, 1)
            init_image_list.append(cur_image)
        init_image_list = torch.cat(init_image_list, dim=0)
        niters = math.ceil(init_image_list.size(0) / batch_size)
        init_image_list = init_image_list.chunk(niters)
        model.register_schedule(given_betas=None, beta_schedule="linear", timesteps=1000,
                            linear_start=0.00085, linear_end=0.0120, cosine_s=8e-3)
        model.num_timesteps = 1000
        sqrt_alphas_cumprod = copy.deepcopy(model.sqrt_alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = copy.deepcopy(model.sqrt_one_minus_alphas_cumprod)
        use_timesteps = set(space_timesteps(1000, [opt.ddpm_steps]))
        last_alpha_cumprod = 1.0
        new_betas = []
        timestep_map = []
        for i, alpha_cumprod in enumerate(model.alphas_cumprod):
            if i in use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                timestep_map.append(i)
        new_betas = [beta.data.cpu().numpy() for beta in new_betas]
        model.register_schedule(given_betas=np.array(new_betas), timesteps=len(new_betas))
        model.num_timesteps = 1000
        model.ori_timesteps = list(use_timesteps)
        model.ori_timesteps.sort()
        model = model.to(device)
        precision_scope = autocast if opt.precision == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    tic = time.time()
                    all_samples = list()
                    for n in trange(niters, desc="Sampling"):
                        init_image = init_image_list[n]
                        init_latent_generator, enc_fea_lq = vq_model.encode(init_image)
                        init_latent = model.get_first_stage_encoding(init_latent_generator)
                        batch,_,_,_=init_latent.shape
                        if model.cond_stage_model.Type == 'semantic':
                            c = resize(init_image)
                            semantic_c = model.cond_stage_model(c)
                            lq = resize(init_image)
                        else:
                            text_init = ['']*init_image.size(0)
                            semantic_c = model.cond_stage_model(text_init)
                            lq = resize(init_image)
                        noise = torch.randn_like(init_latent)
                        # If you would like to start from the intermediate steps, you can add noise to LR to the specific steps.
                        t = repeat(torch.tensor([999]), '1 -> b', b=init_image.size(0))
                        t = t.to(device).long()
                        x_T = model.q_sample_respace(x_start=init_latent, t=t, sqrt_alphas_cumprod=sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, noise=noise)
                        x_T = None
                        task_id = []
                        a= torch.tensor(a)
                        for i in range(batch):
                            task_id.append(a)
                        if model.cond_stage_model.Type == 'semantic':
                            z2 = model.transformer_encoder(init_latent)
                            z2 = model.transformer_decoder(z2)
                            z2 = resize(model.decode_first_stage(z2+init_latent))
                            semantic_c = model.cond_stage_model(z2)
                            samples, _ = model.sample(cond=semantic_c, moe_prompt_cond=[init_latent,task_id,lq], batch_size=init_image.size(0), timesteps=opt.ddpm_steps, time_replace=opt.ddpm_steps, x_T=x_T, return_intermediates=True)
                        else:
                            samples, _ = model.sample(cond=semantic_c, moe_prompt_cond=[init_latent,task_id,lq], batch_size=init_image.size(0), timesteps=opt.ddpm_steps, time_replace=opt.ddpm_steps, x_T=x_T, return_intermediates=True)
                        x_samples = vq_model.decode(samples * 1. / model.scale_factor, enc_fea_lq)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                        for i in range(init_image.size(0)):
                            img_name = img_list.pop(0)
                            basename = os.path.splitext(os.path.basename(img_name))[0]
                            x_sample = 255. * rearrange(x_samples[i].cpu().numpy(), 'c h w -> h w c')
                            Image.fromarray(x_sample.astype(np.uint8)).save(
                                os.path.join(outpath, basename+'.png'))

        print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
            f" \nEnjoy.")


if __name__ == "__main__":
    main()

