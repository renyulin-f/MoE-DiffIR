import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager
import sys
sys.path.insert(0,'../../')
from taming1.taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from timm.models.layers import DropPath, to_2tuple
from ldm.modules.diffusionmodules.model import Encoder, Decoder, Decoder_Mix
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
import torch.nn as nn
from ldm.util import instantiate_from_config

from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop, triplet_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt, random_add_speckle_noise_pt, random_add_saltpepper_noise_pt
import random
import torchvision.transforms as transforms


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            if 'first_stage_model' in k:
                sd[k[18:]] = sd[k]
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Encoder Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def encode(self, x, return_encfea=False):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        if return_encfea:
            return posterior, moments
        return posterior

    def encode_gt(self, x, new_encoder):
        h = new_encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior, moments

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        # x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        x = x.to(memory_format=torch.contiguous_format).float()
        # x = x*2.0-1.0
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            # log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x

# class Prompt(nn.Module):
#     def __init__(self,prompt_len,prompt_channel,prompt_size) -> None:
#         super().__init__()
#         self.prompt_param=nn.Parameter(torch.rand(prompt_len,prompt_channel,prompt_size,prompt_size)-0.5,requires_grad=True)
#         self.prompt_param=torch.nn.init.zeros_(self.prompt_param)
#     def forward(self,x):
#         p = self.prompt_param
#         prompt = p
#         batch,_,_,_=x.shape
#         for i in range(batch-1):
#             prompt=torch.cat((prompt,p),dim=0) 
#         x = x+prompt
#         return x

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops

class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, C, x_size[0],x_size[1]) 
        # x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops    
    
class Prompt(nn.Module):
    def __init__(self,prompt_len,prompt_channel,prompt_size,task_num,top_k) -> None:
        super().__init__()
        self.prompt_len = prompt_len #8
        self.prompt_channel = prompt_channel
        self.prompt_out_size = prompt_size
        self.embed_dim = 256
        self.prompt_param= nn.Parameter(torch.rand(prompt_len,prompt_channel,self.prompt_out_size)-0.5,requires_grad=True)#（8*256*256）
        self.noisy_gating = True
        num_prompts = prompt_len  #8
        self.k = top_k
        self.f_gate = nn.ModuleList([nn.Sequential(               #8个linear层， f_gate[i] = Linear(256,16)
                                        # nn.Linear(input_size, input_size),
                                        # gating_activation,
                                        nn.Linear(prompt_size,
                                                  2 * num_prompts,
                                                  bias=False)
                                    ) for i in range(task_num)])
        
        for i in range(task_num):
            nn.init.zeros_(self.f_gate[i][-1].weight)
        
    @torch.jit.script
    def compute_gating(k: int, probs: torch.Tensor, top_k_gates: torch.Tensor, top_k_indices: torch.Tensor):
        zeros = torch.zeros_like(probs)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        top_k_gates = top_k_gates.flatten()
        top_k_experts = top_k_indices.flatten()
        nonzeros = top_k_gates.nonzero().squeeze(-1)
        top_k_experts_nonzero = top_k_experts[nonzeros]
        _, _index_sorted_experts = top_k_experts_nonzero.sort(0)
        expert_size = (gates > 0).long().sum(0)
        index_sorted_experts = nonzeros[_index_sorted_experts]
        batch_index = index_sorted_experts.div(k, rounding_mode='trunc')
        batch_gates = top_k_gates[index_sorted_experts]
        return batch_gates, batch_index, expert_size, gates, index_sorted_experts
    
    def top_k_gating(self, x, task_bh, bsz,skip_mask=None, sample_topk=0,noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        B = bsz
        index = x.size(0) // B
        result = []
        clean_logits = self.f_gate[task_bh](x)  #(32768,256)
        # for i in range(B):
        #     clean_logits = self.f_gate[task_bh[i]](x[i*index:(i+1)*index,:]) #(1024,32)
        #     result.append(clean_logits)
        # clean_logits = torch.cat(result,dim=0)
        if self.noisy_gating:
            clean_logits, raw_noise_stddev = clean_logits.chunk(2, dim=-1)
            noise_stddev = F.softplus(raw_noise_stddev) + noise_epsilon
            eps = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + eps * noise_stddev
            logits = noisy_logits
        elif self.noisy_gating:
            logits, _ = clean_logits.chunk(2, dim=-1)
        else:
            logits = clean_logits
        probs = torch.softmax(logits, dim=1) + 1e-5 #8192*8
        top_k_gates, top_k_indices = probs.topk(self.k, dim=1) # top_k_gates=(8192,4), top_k_indices=(2,0,1,5)
        #batch_gates = (32768), batch_index =（32768）， expert_size=(8),  gates =(8192,8)  , index_sorted_experts=(32768)
        batch_gates, batch_index, expert_size, gates, index_sorted_experts = \
            self.compute_gating(self.k, probs, top_k_gates, top_k_indices) #4,(8192,4),(8192,4),(2,0,1,5)
        expert_inputs = x[batch_index] #(32768,256)     batch*prompt_len*chossed_expert_num
        num_linears = self.prompt_param.size(0)
        self.batch_index = batch_index
        expert_size_list = expert_size.tolist()
        input_list = expert_inputs.split(expert_size_list, dim=0) #（）
        output_buf_list = []
        for i in range(num_linears):
            output_buf_list.append(torch.mm(input_list[i], self.prompt_param[i]))
        expert_outputs = torch.cat(output_buf_list,dim=0)
        expert_outputs = expert_outputs * batch_gates[:, None]
        return expert_outputs
    
    def forward(self,x,task_id):
        # B = x.size(0)  
        bsz, length, emb_size = x.size() #8,256,1024    (1,1024,256)
        x2 = x.reshape(-1, emb_size)  #(1024,256)
        expert_outputs = self.top_k_gating(x2,task_id,bsz)
        zeros = torch.zeros((bsz * length, emb_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)  #(8192,256)
        y = zeros.index_add(0, self.batch_index, expert_outputs) #(8192,256)
        y = y.view(bsz, length, emb_size) #(8,1024,256)
      
        return y

    
    
class AutoencoderKLResi(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 fusion_w=1.0,
                 freeze_dec=True,
                 synthesis_data=False,
                 use_usm=False,
                 test_gt=False,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder_Mix(**ddconfig)
        self.patch_embed1 = PatchEmbed(
            img_size=256, patch_size=4, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=None
        )
        self.patch_unembed1 = PatchUnEmbed(
            img_size=256, patch_size=4, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=None
        )
        self.Prompt_model1 = Prompt(7,256,256,21,3)
        self.Prompt_model2 = Prompt(7,512,512,21,3)
        self.decoder.fusion_w = fusion_w
        self.loss = instantiate_from_config(lossconfig)
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            missing_list = self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        else:
            missing_list = []

        print('>>>>>>>>>>>>>>>>>missing>>>>>>>>>>>>>>>>>>>')
        print(missing_list)
        self.synthesis_data = synthesis_data
        self.use_usm = use_usm
        self.test_gt = test_gt

        if freeze_dec:
            for name, param in self.named_parameters():
                if 'fusion_layer' in name:
                    param.requires_grad = True
                # elif 'encoder' in name:
                #     param.requires_grad = True
                # elif 'quant_conv' in name and 'post_quant_conv' not in name:
                #     param.requires_grad = True
                elif 'loss.discriminator' in name:
                    param.requires_grad = True
                elif 'Prompt' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        print('>>>>>>>>>>>>>>>>>trainable_list>>>>>>>>>>>>>>>>>>>')
        trainable_list = []
        for name, params in self.named_parameters():
            if params.requires_grad:
                trainable_list.append(name)
        print(trainable_list)

        print('>>>>>>>>>>>>>>>>>Untrainable_list>>>>>>>>>>>>>>>>>>>')
        untrainable_list = []
        for name, params in self.named_parameters():
            if not params.requires_grad:
                untrainable_list.append(name)
        print(untrainable_list)
        # untrainable_list = list(set(trainable_list).difference(set(missing_list)))
        # print('>>>>>>>>>>>>>>>>>untrainable_list>>>>>>>>>>>>>>>>>>>')
        # print(untrainable_list)

    # def init_from_ckpt(self, path, ignore_keys=list()):
    #     sd = torch.load(path, map_location="cpu")["state_dict"]
    #     keys = list(sd.keys())
    #     for k in keys:
    #         for ik in ignore_keys:
    #             if k.startswith(ik):
    #                 print("Deleting key {} from state_dict.".format(k))
    #                 del sd[k]
    #     self.load_state_dict(sd, strict=False)
    #     print(f"Restored from {path}")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            if 'first_stage_model' in k:
                sd[k[18:]] = sd[k]
                del sd[k]
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Encoder Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
        return missing

    def encode(self, x):
        h, enc_fea = self.encoder(x, return_fea=True)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        # posterior = h
        return posterior, enc_fea

    def encode_gt(self, x, new_encoder):
        h = new_encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior, moments

    def decode(self, z, enc_fea):
        z = self.post_quant_conv(z)
        dec = self.decoder(z, enc_fea)
        return dec

    def forward(self, input, latent, sample_posterior=True):
        posterior, enc_fea_lq = self.encode(input)
        x1 = self.Prompt_model1(self.patch_embed1(enc_fea_lq[0]),1)
        enc_fea_lq[0] = self.patch_unembed1(x1,(128,128))
        x2 = self.Prompt_model2(self.patch_embed1(enc_fea_lq[1]),1)
        enc_fea_lq[1] = self.patch_unembed1(x2,(64,64))
        # enc_fea_lq[0] = self.Prompt_model1(enc_fea_lq[0])   #(1,256,128)  (2,256,128,128)
        # enc_fea_lq[1] = self.Prompt_model2(enc_fea_lq[1])   (2,512,64,64)
        dec = self.decode(latent, enc_fea_lq)
        return dec, posterior

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        _, c_, h_, w_ = self.latent.size()
        if b == self.configs.data.params.batch_size:
            if not hasattr(self, 'queue_size'):
                self.queue_size = self.configs.data.params.train.params.get('queue_size', b*50)
            if not hasattr(self, 'queue_lr'):
                assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
                self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
                _, c, h, w = self.gt.size()
                self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
                self.queue_sample = torch.zeros(self.queue_size, c, h, w).cuda()
                self.queue_latent = torch.zeros(self.queue_size, c_, h_, w_).cuda()
                self.queue_ptr = 0
            if self.queue_ptr == self.queue_size:  # the pool is full
                # do dequeue and enqueue
                # shuffle
                idx = torch.randperm(self.queue_size)
                self.queue_lr = self.queue_lr[idx]
                self.queue_gt = self.queue_gt[idx]
                self.queue_sample = self.queue_sample[idx]
                self.queue_latent = self.queue_latent[idx]
                # get first b samples
                lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
                gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
                sample_dequeue = self.queue_sample[0:b, :, :, :].clone()
                latent_dequeue = self.queue_latent[0:b, :, :, :].clone()
                # update the queue
                self.queue_lr[0:b, :, :, :] = self.lq.clone()
                self.queue_gt[0:b, :, :, :] = self.gt.clone()
                self.queue_sample[0:b, :, :, :] = self.sample.clone()
                self.queue_latent[0:b, :, :, :] = self.latent.clone()

                self.lq = lq_dequeue
                self.gt = gt_dequeue
                self.sample = sample_dequeue
                self.latent = latent_dequeue
            else:
                # only do enqueue
                self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
                self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
                self.queue_sample[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.sample.clone()
                self.queue_latent[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.latent.clone()
                self.queue_ptr = self.queue_ptr + b

    def get_input(self, batch):
        input = batch['lq']
        gt = batch['gt']
        latent = batch['latent']
        sample = batch['sample']

        assert not torch.isnan(latent).any()

        input = input.to(memory_format=torch.contiguous_format).float()
        gt = gt.to(memory_format=torch.contiguous_format).float()
        latent = latent.to(memory_format=torch.contiguous_format).float() / 0.18215

        gt = gt / 255.0
        input = input / 255.0
        sample = sample / 255.0
        
        gt = gt * 2.0 - 1.0
        input = input * 2.0 - 1.0
        sample = sample * 2.0 -1.0

        return input, gt, latent, sample

    @torch.no_grad()
    def get_input_synthesis(self, batch, val=False, test_gt=False):
        im_gt = batch['gt'].cuda()
        if self.use_usm:
            usm_sharpener = USMSharp().cuda()  # do usm sharpening
            im_gt = usm_sharpener(im_gt)
        im_gt = im_gt.to(memory_format=torch.contiguous_format).float()
        im_lq = batch['lq'].cuda()
        self.lq, self.gt = im_lq, im_gt
        self.latent = batch['latent'] / 0.18215
        self.sample = batch['sample'] * 2 - 1.0
        if not val:
            self._dequeue_and_enqueue()
        # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
        self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
        self.lq = self.lq*2 - 1.0
        self.gt = self.gt*2 - 1.0
        self.lq = torch.clamp(self.lq, -1.0, 1.0)
        x = self.lq
        y = self.gt
        x = x.to(self.device)
        y = y.to(self.device)
        if self.test_gt:
            return y, y, self.latent.to(self.device), self.sample.to(self.device)
        else:
            return x, y, self.latent.to(self.device), self.sample.to(self.device)

    def training_step(self, batch, batch_idx, optimizer_idx):
        if self.synthesis_data:
            inputs, gts, latents, _ = self.get_input_synthesis(batch, val=False)
        else:
            inputs, gts, latents, _ = self.get_input(batch)
        reconstructions, posterior = self(inputs, latents)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(gts, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(gts, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs, gts, latents, _ = self.get_input(batch)

        reconstructions, posterior = self(inputs, latents)
        aeloss, log_dict_ae = self.loss(gts, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(gts, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  # list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        if self.synthesis_data:
            x, gts, latents, samples = self.get_input_synthesis(batch, val=False)
        else:
            x, gts, latents, samples = self.get_input(batch)
        x = x.to(self.device)
        latents = latents.to(self.device)
        samples = samples.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x, latents)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                gts = self.to_rgb(gts)
                samples = self.to_rgb(samples)
                xrec = self.to_rgb(xrec)
            # log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        log["gts"] = gts
        log["samples"] = samples
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
    
    
class AutoencoderKLResi2(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 fusion_w=1.0,
                 freeze_dec=True,
                 synthesis_data=False,
                 use_usm=False,
                 test_gt=False,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder_Mix(**ddconfig)
        # self.Prompt_model1 = Prompt(1,256,128)
        # self.Prompt_model2 = Prompt(1,512,64)
        self.decoder.fusion_w = fusion_w
        self.loss = instantiate_from_config(lossconfig)
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            missing_list = self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        else:
            missing_list = []

        print('>>>>>>>>>>>>>>>>>missing>>>>>>>>>>>>>>>>>>>')
        print(missing_list)
        self.synthesis_data = synthesis_data
        self.use_usm = use_usm
        self.test_gt = test_gt

        if freeze_dec:
            for name, param in self.named_parameters():
                if 'fusion_layer' in name:
                    param.requires_grad = True
                # elif 'encoder' in name:
                #     param.requires_grad = True
                # elif 'quant_conv' in name and 'post_quant_conv' not in name:
                #     param.requires_grad = True
                elif 'loss.discriminator' in name:
                    param.requires_grad = True
                elif 'Prompt' in name:
                    print('Prompt')
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        print('>>>>>>>>>>>>>>>>>trainable_list>>>>>>>>>>>>>>>>>>>')
        trainable_list = []
        for name, params in self.named_parameters():
            if params.requires_grad:
                trainable_list.append(name)
        print(trainable_list)

        print('>>>>>>>>>>>>>>>>>Untrainable_list>>>>>>>>>>>>>>>>>>>')
        untrainable_list = []
        for name, params in self.named_parameters():
            if not params.requires_grad:
                untrainable_list.append(name)
        print(untrainable_list)
        # untrainable_list = list(set(trainable_list).difference(set(missing_list)))
        # print('>>>>>>>>>>>>>>>>>untrainable_list>>>>>>>>>>>>>>>>>>>')
        # print(untrainable_list)

    # def init_from_ckpt(self, path, ignore_keys=list()):
    #     sd = torch.load(path, map_location="cpu")["state_dict"]
    #     keys = list(sd.keys())
    #     for k in keys:
    #         for ik in ignore_keys:
    #             if k.startswith(ik):
    #                 print("Deleting key {} from state_dict.".format(k))
    #                 del sd[k]
    #     self.load_state_dict(sd, strict=False)
    #     print(f"Restored from {path}")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            if 'first_stage_model' in k:
                sd[k[18:]] = sd[k]
                del sd[k]
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Encoder Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
        return missing

    def encode(self, x):
        h, enc_fea = self.encoder(x, return_fea=True)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        # posterior = h
        return posterior, enc_fea

    def encode_gt(self, x, new_encoder):
        h = new_encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior, moments

    def decode(self, z, enc_fea):
        z = self.post_quant_conv(z)
        dec = self.decoder(z, enc_fea)
        return dec

    def forward(self, input, latent, sample_posterior=True):
        posterior, enc_fea_lq = self.encode(input)
        # enc_fea_lq[0] = self.Prompt_model1(enc_fea_lq[0])
        # enc_fea_lq[1] = self.Prompt_model2(enc_fea_lq[1])
        dec = self.decode(latent, enc_fea_lq)
        return dec, posterior

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        _, c_, h_, w_ = self.latent.size()
        if b == self.configs.data.params.batch_size:
            if not hasattr(self, 'queue_size'):
                self.queue_size = self.configs.data.params.train.params.get('queue_size', b*50)
            if not hasattr(self, 'queue_lr'):
                assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
                self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
                _, c, h, w = self.gt.size()
                self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
                self.queue_sample = torch.zeros(self.queue_size, c, h, w).cuda()
                self.queue_latent = torch.zeros(self.queue_size, c_, h_, w_).cuda()
                self.queue_ptr = 0
            if self.queue_ptr == self.queue_size:  # the pool is full
                # do dequeue and enqueue
                # shuffle
                idx = torch.randperm(self.queue_size)
                self.queue_lr = self.queue_lr[idx]
                self.queue_gt = self.queue_gt[idx]
                self.queue_sample = self.queue_sample[idx]
                self.queue_latent = self.queue_latent[idx]
                # get first b samples
                lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
                gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
                sample_dequeue = self.queue_sample[0:b, :, :, :].clone()
                latent_dequeue = self.queue_latent[0:b, :, :, :].clone()
                # update the queue
                self.queue_lr[0:b, :, :, :] = self.lq.clone()
                self.queue_gt[0:b, :, :, :] = self.gt.clone()
                self.queue_sample[0:b, :, :, :] = self.sample.clone()
                self.queue_latent[0:b, :, :, :] = self.latent.clone()

                self.lq = lq_dequeue
                self.gt = gt_dequeue
                self.sample = sample_dequeue
                self.latent = latent_dequeue
            else:
                # only do enqueue
                self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
                self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
                self.queue_sample[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.sample.clone()
                self.queue_latent[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.latent.clone()
                self.queue_ptr = self.queue_ptr + b

    def get_input(self, batch):
        input = batch['lq']
        gt = batch['gt']
        latent = batch['latent']
        sample = batch['sample']

        assert not torch.isnan(latent).any()

        input = input.to(memory_format=torch.contiguous_format).float()
        gt = gt.to(memory_format=torch.contiguous_format).float()
        latent = latent.to(memory_format=torch.contiguous_format).float() / 0.18215

        gt = gt * 2.0 - 1.0
        input = input * 2.0 - 1.0
        sample = sample * 2.0 -1.0

        return input, gt, latent, sample

    @torch.no_grad()
    def get_input_synthesis(self, batch, val=False, test_gt=False):

        jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        im_gt = batch['gt'].cuda()
        if self.use_usm:
            usm_sharpener = USMSharp().cuda()  # do usm sharpening
            im_gt = usm_sharpener(im_gt)
        im_gt = im_gt.to(memory_format=torch.contiguous_format).float()
        kernel1 = batch['kernel1'].cuda()
        kernel2 = batch['kernel2'].cuda()
        sinc_kernel = batch['sinc_kernel'].cuda()

        ori_h, ori_w = im_gt.size()[2:4]

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(im_gt, kernel1)
        # random resize
        updown_type = random.choices(
                ['up', 'down', 'keep'],
                self.configs.degradation['resize_prob'],
                )[0]
        if updown_type == 'up':
            scale = random.uniform(1, self.configs.degradation['resize_range'][1])
        elif updown_type == 'down':
            scale = random.uniform(self.configs.degradation['resize_range'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        # add noise
        gray_noise_prob = self.configs.degradation['gray_noise_prob']
        if random.random() < self.configs.degradation['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.configs.degradation['noise_range'],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
                )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.configs.degradation['poisson_scale_range'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range'])
        out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if random.random() < self.configs.degradation['second_blur_prob']:
            out = filter2D(out, kernel2)
        # random resize
        updown_type = random.choices(
                ['up', 'down', 'keep'],
                self.configs.degradation['resize_prob2'],
                )[0]
        if updown_type == 'up':
            scale = random.uniform(1, self.configs.degradation['resize_range2'][1])
        elif updown_type == 'down':
            scale = random.uniform(self.configs.degradation['resize_range2'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
                out,
                size=(int(ori_h / self.configs.sf * scale),
                      int(ori_w / self.configs.sf * scale)),
                mode=mode,
                )
        # add noise
        gray_noise_prob = self.configs.degradation['gray_noise_prob2']
        if random.random() < self.configs.degradation['gaussian_noise_prob2']:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.configs.degradation['noise_range2'],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
                )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.configs.degradation['poisson_scale_range2'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False,
                )
            
        if random.random() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                    out,
                    size=(ori_h // self.configs.sf,
                          ori_w // self.configs.sf),
                    mode=mode,
                    )
            out = filter2D(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                    out,
                    size=(ori_h // self.configs.sf,
                          ori_w // self.configs.sf),
                    mode=mode,
                    )
            out = filter2D(out, sinc_kernel)

        # clamp and round
        im_lq = torch.clamp(out, 0, 1.0)

        # random crop
        gt_size = self.configs.degradation['gt_size']
        im_gt, im_lq = paired_random_crop(im_gt, im_lq, gt_size, self.configs.sf)
        self.lq, self.gt = im_lq, im_gt

        self.lq = F.interpolate(
                self.lq,
                size=(self.gt.size(-2),
                      self.gt.size(-1)),
                mode='bicubic',
                )

        self.latent = batch['latent'] / 0.18215
        self.sample = batch['sample'] * 2 - 1.0
        # training pair pool
        if not val:
            self._dequeue_and_enqueue()
        # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
        self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
        self.lq = self.lq*2 - 1.0
        self.gt = self.gt*2 - 1.0

        self.lq = torch.clamp(self.lq, -1.0, 1.0)

        x = self.lq
        y = self.gt
        x = x.to(self.device)
        y = y.to(self.device)

        if self.test_gt:
            return y, y, self.latent.to(self.device), self.sample.to(self.device)
        else:
            return x, y, self.latent.to(self.device), self.sample.to(self.device)

    def training_step(self, batch, batch_idx, optimizer_idx):
        if self.synthesis_data:
            inputs, gts, latents, _ = self.get_input_synthesis(batch, val=False)
        else:
            inputs, gts, latents, _ = self.get_input(batch)
        reconstructions, posterior = self(inputs, latents)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(gts, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(gts, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    # def validation_step(self, batch, batch_idx):
    #     inputs, gts, latents, _ = self.get_input(batch)

    #     reconstructions, posterior = self(inputs, latents)
    #     aeloss, log_dict_ae = self.loss(gts, reconstructions, posterior, 0, self.global_step,
    #                                     last_layer=self.get_last_layer(), split="val")

    #     discloss, log_dict_disc = self.loss(gts, reconstructions, posterior, 1, self.global_step,
    #                                         last_layer=self.get_last_layer(), split="val")

    #     self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
    #     self.log_dict(log_dict_ae)
    #     self.log_dict(log_dict_disc)
    #     return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  # list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        if self.synthesis_data:
            x, gts, latents, samples = self.get_input_synthesis(batch, val=False)
        else:
            x, gts, latents, samples = self.get_input(batch)
        x = x.to(self.device)
        latents = latents.to(self.device)
        samples = samples.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x, latents)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                gts = self.to_rgb(gts)
                samples = self.to_rgb(samples)
                xrec = self.to_rgb(xrec)
            # log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        log["gts"] = gts
        log["samples"] = samples
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
