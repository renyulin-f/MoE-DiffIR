from abc import abstractmethod
import math
import torch
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple
from einops import rearrange, repeat
from torch import einsum
import sys
sys.path.append('../../')
import open_clip


try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from ldm.modules.attention import SpatialTransformer, SpatialTransformerV2
from ldm.modules.spade import SPADE


def default(val, d):
    return val if exists(val) else d

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        #query_dim=320 ,inner_dim=320, context_dim=320
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)

        k = self.to_k(context) #(3,7,320)
        v = self.to_v(context)  #(3,7,64*5)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v)) #(15,1024,64)
        #到这里应该是从(3,1024,320)->(15,1024,64)
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        #sim=(15,1024,1024)
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)  #(3,1024,320)



class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, head_dim=None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads #8
        if head_dim is None:
            head_dim = dim // num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(inner_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        if torch.isinf(attn).any():
            clamp_value = torch.finfo(attn.dtype).max-1000
            attn = torch.clamp(attn, min=-clamp_value, max=clamp_value)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):  #相当于是feedforward 层
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class MoEnhanceTaskBlock(nn.Module):

    def __init__(self, dim, num_heads,  qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)#nn.Layernorm(256)
        self.attn = Attention(dim, num_heads=num_heads, head_dim=head_dim, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, hidden_features=dim * 4, drop=drop)  #dim=768, hidden_features= 768*4,drop=0.0

    def forward(self, x, task_bh, mask=None):
    
        y = self.attn(self.norm1(x), mask=mask)
        x = x + self.drop_path(y) #(1,197,768)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



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
    
class SinglePrompt(nn.Module):
    def __init__(self,prompt_channel,prompt_size) -> None:
        super().__init__()
        self.prompt_channel = prompt_channel
        self.prompt_out_size = prompt_size
        self.prompt_param = nn.Parameter(torch.rand(prompt_channel,self.prompt_out_size)-0.5,requires_grad=True)
        self.prompt_param =torch.nn.init.zeros_(self.prompt_param)
            
    def forward(self,x,task_id):
        bsz, length, emb_size = x.size() #8,256,1024    (1,1024,256)
        x = x.reshape(-1, emb_size)  #(1024,256)
        expert_inputs = x
        expert_outputs = (torch.mm(expert_inputs, self.prompt_param))
        y = expert_outputs.view(bsz, length, emb_size) #(8,1024,256)
        return y
    
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
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0],x_size[1]) 
        return x

    def flops(self):
        flops = 0
        return flops
    
class MoE_Prompt(nn.Module):
    def __init__(self,prompt_len,prompt_channel,prompt_size,task_num,top_k) -> None:
        super().__init__()
        self.prompt_len = prompt_len #8
        self.prompt_channel = prompt_channel
        self.prompt_out_size = prompt_size
        self.embed_dim = 256
        self.prompt_param = nn.Parameter(torch.rand(prompt_len,prompt_channel,self.prompt_out_size)-0.5,requires_grad=True)#（8*256*256）
        self.noisy_gating = True
        num_prompts = prompt_len  #8
        input_size = 256
        self.k = top_k
        self.f_gate = nn.ModuleList([nn.Sequential(               #8个linear层， f_gate[i] = Linear(256,16)
                                        nn.Linear(input_size,
                                                  2 * num_prompts,
                                                  bias=False)
                                    ) for _ in range(task_num)])
        
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
    
    def top_k_gating(self, x, task_bh, bsz,noise_epsilon=1e-2):
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
        for i in range(B):
            clean_logits = self.f_gate[task_bh[i]](x[i*index:(i+1)*index,:]) #(1024,32)
            result.append(clean_logits)
        clean_logits = torch.cat(result,dim=0)
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
        probs = torch.softmax(logits, dim=1) + 1e-5
        top_k_gates, top_k_indices = probs.topk(self.k, dim=1) 
        batch_gates, batch_index, expert_size, gates, index_sorted_experts = \
            self.compute_gating(self.k, probs, top_k_gates, top_k_indices) 
        expert_inputs = x[batch_index] #batch*prompt_len*chossed_expert_num
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
        bsz, length, emb_size = x.size() 
        x2 = x.reshape(-1, emb_size)  
        expert_outputs = self.top_k_gating(x2,task_id,bsz)
        zeros = torch.zeros((bsz * length, emb_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device) 
        y = zeros.index_add(0, self.batch_index, expert_outputs) 
        y = y.view(bsz, length, emb_size)
      
        return y


    
class Prompt_routing(nn.Module):
    def __init__(self,prompt_len,prompt_channel,prompt_size,task_num,top_k) -> None:
        super().__init__()
        self.prompt_len = prompt_len #8
        self.prompt_channel = prompt_channel
        self.prompt_out_size = prompt_size
        self.embed_dim = 256
        self.attn = CrossAttention(self.embed_dim,768)#DA-CLIP
        self.prompt_param= nn.Parameter(torch.rand(prompt_len,prompt_channel,self.prompt_out_size)-0.5,requires_grad=True)#（8*256*256）
        self.noisy_gating = True
        num_prompts = prompt_len  #8
        input_size = 256
        self.k = top_k
        self.f_gate = nn.ModuleList([nn.Sequential(               #8 linear layer， f_gate[i] = Linear(256,16)
                                        nn.Linear(input_size,
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
    
    def top_k_gating(self, x_old, task_bh, bsz,degradation_prior, noise_epsilon=1e-2):
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
        x = degradation_prior
        B = bsz
        index = x.size(0) // B
        result = []
        for i in range(B):
            clean_logits = self.f_gate[task_bh[i]](x[i*index:(i+1)*index,:]) #(1024,32)
            result.append(clean_logits)
        clean_logits = torch.cat(result,dim=0)
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
        batch_gates, batch_index, expert_size, gates, index_sorted_experts = \
            self.compute_gating(self.k, probs, top_k_gates, top_k_indices) #4,(8192,4),(8192,4),(2,0,1,5)
        
        expert_inputs = x_old[batch_index] #(32768,256)     batch*prompt_len*chossed_expert_num
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
    
    def forward(self,x,task_id,inter):
        x_old = x
        bsz, length, emb_size = x.size() #8,256,1024    (1,1024,256)
        inter = rearrange(inter, 'b c h w -> b (h w) c').contiguous() 
        degradation_prior = self.attn(x,inter) #（4，256，32，32）， inter= (4,64,256,256)
        x_old = x_old.reshape(-1, emb_size)  #(1024,256)
        x_new = degradation_prior.reshape(-1,emb_size)
        expert_outputs = self.top_k_gating(x_old,task_id,bsz,x_new)

        zeros = torch.zeros((bsz * length, emb_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)  #(8192,256)
        y = zeros.index_add(0, self.batch_index, expert_outputs) #(8192,256)
        y = y.view(bsz, length, emb_size) #(8,1024,256)
      
        return y
    
class WeightedPrompt(nn.Module):
    def __init__(self,prompt_len,prompt_channel,prompt_size) -> None:
        super().__init__()
        self.prompt_len = prompt_len
        self.prompt_channel = prompt_channel
        self.prompt_out_size = prompt_size
        self.prompt_param = nn.Parameter(torch.rand(prompt_len,self.prompt_out_size)-0.5,requires_grad=True)#(21,256,256)
        self.linear_layer = nn.Linear(256,prompt_len)
        self.prompt_param =torch.nn.init.zeros_(self.prompt_param)
            
    def forward(self,x,task_id):
        bsz, length, emb_size = x.size() #8,256,1024    (1,1024,256)
        x = x.reshape(-1, emb_size)  #(1024,256)
        prompt_weights = F.softmax(self.linear_layer(x),dim=1) #(1024,21)
        prompts = torch.mm(prompt_weights,self.prompt_param)
        expert_inputs = x #(1024,256)
        expert_outputs = expert_inputs+prompts#(1024,256)
        y = expert_outputs.view(bsz, length, emb_size) #(8,1024,256)
        return y
    
    
class MOETaskTransformer(nn.Module):
    """
    The half UNet model with attention and timestep embedding.
    For usage, see UNet.
    """

    def __init__(
        self, img_types, embed_dim=256, depth=2, num_heads=4, norm_layer=nn.LayerNorm, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., qkv_bias=True, 
                 head_dim=None, top_k=1, num_prompt_experts=7,
                 use_moe_prompt=False, use_single_prompt=False,
                 use_DACLIP_prior = False, use_weighted_prompt = False,
                 prompt_dim=256, prompt_size=256
                 
    ):
        super().__init__()
        self.use_DACLIP_prior = use_DACLIP_prior
        self.use_weighted_prompt = use_weighted_prompt
        self.img_types = img_types #['jpeg','vvc','hevc','webp'...]
        self.depth = depth
        self.taskGating = True
        self.top_k = top_k
        self.use_moe_prompt = use_moe_prompt
        self.use_single_prompt = use_single_prompt
        
        # Wheather use DACLIP-Prior
        if self.use_DACLIP_prior:
            checkpoint = 'checkpoints_degradation/daclip_ViT-B-32.pt'
            self.DACLIP_model, self.preprocess = open_clip.create_model_from_pretrained('daclip_ViT-B-32', pretrained=checkpoint)
            self.DACLIP_model.eval()
            for _ , param in self.DACLIP_model.named_parameters():
                param.requires_grad = False
                
        self.task_num = len(self.img_types)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        conv_resample = True
        dims = 2
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.patch_embed1 = PatchEmbed(
            img_size=256, patch_size=4, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=None
        )
        self.norm1 = norm_layer(embed_dim)
        self.patch_unembed1 = PatchUnEmbed(
            img_size=256, patch_size=4, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=None
        )
        
        self.patch_embed2 = PatchEmbed(
            img_size=256, patch_size=4, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=None
        )
        self.norm2 = norm_layer(embed_dim)
        self.patch_unembed2 = PatchUnEmbed(
            img_size=256, patch_size=4, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=None
        )
        
        self.patch_embed3 = PatchEmbed(
            img_size=256, patch_size=4, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=None
        )
        self.norm3 = norm_layer(embed_dim)
        self.patch_unembed3 = PatchUnEmbed(
            img_size=256, patch_size=4, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=None
        )
        
        self.patch_embed4 = PatchEmbed(
            img_size=256, patch_size=4, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=None
        )
        self.norm4 = norm_layer(embed_dim)
        self.patch_unembed4 = PatchUnEmbed(
            img_size=256, patch_size=4, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=None
        )
        
        # The dimension of the Prompt: (Num_prompt, Prompt_dim, Prompt_size)
        if self.use_moe_prompt:
            if self.use_DACLIP_prior:
                self.Prompt_model1 = Prompt_routing(num_prompt_experts,prompt_dim,prompt_size,self.task_num,self.top_k)
                self.Prompt_model2 = Prompt_routing(num_prompt_experts,prompt_dim,prompt_size,self.task_num,self.top_k)
                self.Prompt_model3 = Prompt_routing(num_prompt_experts,prompt_dim,prompt_size,self.task_num,self.top_k)
                self.Prompt_model4 = Prompt_routing(num_prompt_experts,prompt_dim,prompt_size,self.task_num,self.top_k)
            else:
                self.Prompt_model1 = MoE_Prompt(num_prompt_experts,prompt_dim,prompt_size,self.task_num,self.top_k) 
                self.Prompt_model2 = MoE_Prompt(num_prompt_experts,prompt_dim,prompt_size,self.task_num,self.top_k)
                self.Prompt_model3 = MoE_Prompt(num_prompt_experts,prompt_dim,prompt_size,self.task_num,self.top_k)
                self.Prompt_model4 = MoE_Prompt(num_prompt_experts,prompt_dim,prompt_size,self.task_num,self.top_k)            
        
        if self.use_single_prompt:
            self.Prompt_model = SinglePrompt(prompt_dim,prompt_size)
            
        if self.use_weighted_prompt:
            self.Prompt_model1 = WeightedPrompt(self.task_num,prompt_dim,prompt_size)
            self.Prompt_model2 = WeightedPrompt(self.task_num,prompt_dim,prompt_size)
            self.Prompt_model3 = WeightedPrompt(self.task_num,prompt_dim,prompt_size)
            self.Prompt_model4 = WeightedPrompt(self.task_num,prompt_dim,prompt_size)
            
        self.conv_first = nn.Sequential(
                nn.Conv2d(4, embed_dim, 3, 1, 1),
            )

        self.blocks1 = nn.Sequential(*[
            MoEnhanceTaskBlock(
                head_dim=head_dim, #num_experts=4
                dim=embed_dim, num_heads=num_heads,  qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                )
            for i in range(depth)])
        
        ch = embed_dim
        out_ch = ch
        self.downsample1 = Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
        
        self.blocks2 = nn.Sequential(*[
            MoEnhanceTaskBlock(
                head_dim=head_dim, #num_experts=4
                dim=embed_dim, num_heads=num_heads,  qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                )
            for i in range(depth)])
        self.downsample2 = Downsample(
                        ch, conv_resample, dims=dims, out_channels=out_ch
                    )
        
        self.blocks3 = nn.Sequential(*[
            MoEnhanceTaskBlock(
                head_dim=head_dim, #num_experts=4
                dim=embed_dim, num_heads=num_heads,  qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                )
            for i in range(depth)])
        self.downsample3 = Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
        
        self.blocks4 = nn.Sequential(*[
            MoEnhanceTaskBlock(
                head_dim=head_dim, #num_experts=4
                dim=embed_dim, num_heads=num_heads,  qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                )
            for i in range(depth)])
    
    def forward(self, x, task_id,lq):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        result_list = []
        results = {}
        x = self.conv_first(x) #(1,256,32,32)
        x = self.patch_embed1(x) #(1,256,32,32)->(1,1024,256)
        x_before = self.pos_drop(x)
        x = x_before
        t = task_id
        
        if self.use_DACLIP_prior:
            image = lq
            degra_features = self.DACLIP_model.encode_image(image, control=True)
            degra_features /= degra_features.norm(dim=-1, keepdim=True)
            inter = degra_features

        for blk in self.blocks1:
            x = blk(x, t) #t = task_index
        x = self.norm1(x)
        
        if self.use_moe_prompt:
            if self.use_DACLIP_prior:
                out1 = self.patch_unembed1(x,(32,32))
                x2 = self.Prompt_model1(x,t,inter)
            else:
                out1 = self.patch_unembed1(x,(32,32))
                x2 = self.Prompt_model1(x,t)
        if self.use_single_prompt: 
            x2 = self.Prompt_model(x,t)
            out1 = self.patch_unembed1(x,(32,32))
        if self.use_weighted_prompt:
            x2 = self.Prompt_model1(x,t)
            out1 = self.patch_unembed1(x,(32,32))
            
        out2 = self.patch_unembed1(x2,(32,32)) #(1,160,32,32)
        result_list.append(out2)
        x = self.downsample1(out1) #(1,256,16,16)
        
    # Second Layer
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x, t) #t = task_index
        x = self.norm2(x)
        if self.use_moe_prompt:
            if self.use_DACLIP_prior:
                x2 = self.Prompt_model2(x,t,inter)
                out1 = self.patch_unembed2(x,(16,16))
            else:
                x2 = self.Prompt_model2(x,t)
                out1 = self.patch_unembed2(x,(16,16))
        if self.use_single_prompt:
            x2 = self.Prompt_model(x,t)
            out1 = self.patch_unembed2(x,(16,16))
        if self.use_weighted_prompt:
            x2 = self.Prompt_model2(x,t)
            out1 = self.patch_unembed2(x,(16,16))

        out2 = self.patch_unembed2(x2,(16,16)) #(1,160,32,32)
        result_list.append(out2)
        x = self.downsample2(out1)
        
    # Third Layer
        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x, t) #t = task_index
        x = self.norm3(x)
        if self.use_moe_prompt:
            if self.use_DACLIP_prior:
                x2 = self.Prompt_model3(x,t,inter)
                out1 = self.patch_unembed3(x,(8,8))
            else:
                x2 = self.Prompt_model3(x,t)
                out1 = self.patch_unembed3(x,(8,8))
        if self.use_single_prompt:
            x2 = self.Prompt_model(x,t)
            out1 = self.patch_unembed3(x,(8,8))
        if self.use_weighted_prompt:
            x2 = self.Prompt_model3(x,t)
            out1 = self.patch_unembed3(x,(8,8))
        out2 = self.patch_unembed3(x2,(8,8)) #(1,160,32,32)
        result_list.append(out2)
        x = self.downsample3(out1)
        
    #Fourth Layer
        x = self.patch_embed4(x)
        for blk in self.blocks4:
            x = blk(x, t) #t = task_index
        x2 = self.norm4(x)
        if self.use_moe_prompt:
            if self.use_DACLIP_prior:
                x2 = self.Prompt_model4(x,t,inter)
            else:
                x2 = self.Prompt_model4(x,t)
        if self.use_single_prompt:
            x2 = self.Prompt_model(x,t)
        if self.use_weighted_prompt:
            x2 = self.Prompt_model4(x,t)
        out2 = self.patch_unembed4(x2,(4,4)) #(1,160,32,32)
        result_list.append(out2)
        for i in range(len(result_list)):
            results[str(result_list[i].size(-1))] = result_list[i]
    
        return results


# dummy replace
def convert_module_to_f16(x):
    pass

def convert_module_to_f32(x):
    pass

def exists(val):
    return val is not None

def cal_fea_cossim(fea_1, fea_2, save_dir=None):
    cossim_fuc = nn.CosineSimilarity(dim=-1, eps=1e-6)
    if save_dir is None:
        save_dir_1 = './cos_sim64_1_not.txt'
        save_dir_2 = './cos_sim64_2_not.txt'
    b, c, h, w = fea_1.size()
    fea_1 = fea_1.reshape(b, c, h*w)
    fea_2 = fea_2.reshape(b, c, h*w)
    cos_sim = cossim_fuc(fea_1, fea_2)
    cos_sim = cos_sim.data.cpu().numpy()
    with open(save_dir_1, "a") as my_file:
        my_file.write(str(np.mean(cos_sim[0])) + "\n")

## go
class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5)
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepBlockDual(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb, cond):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepBlock3cond(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb, s_cond, seg_cond):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None, moe_prompt_cond=None, seg_cond=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer) or isinstance(layer, SpatialTransformerV2):
                assert context is not None
                x = layer(x, context)
            elif isinstance(layer, TimestepBlockDual):
                assert moe_prompt_cond is not None
                x = layer(x, emb, moe_prompt_cond)
            elif isinstance(layer, TimestepBlock3cond):
                assert seg_cond is not None
                x = layer(x, emb, moe_prompt_cond, seg_cond)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class TransposedUpsample(nn.Module):
    'Learned 2x upsampling without padding'
    def __init__(self, channels, out_channels=None, ks=5):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.up = nn.ConvTranspose2d(self.channels,self.out_channels,kernel_size=ks,stride=2)

    def forward(self,x):
        return self.up(x)


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        if self.out_channels % 32 == 0:
            self.out_layers = nn.Sequential(
                normalization(self.out_channels),
                nn.SiLU(),
                nn.Dropout(p=dropout),
                zero_module(
                    conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
                ),
            )
        else:
            self.out_layers = nn.Sequential(
                normalization(self.out_channels, self.out_channels),
                nn.SiLU(),
                nn.Dropout(p=dropout),
                zero_module(
                    conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
                ),
            )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )


    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class ResBlockDual(TimestepBlockDual):
    """
    A residual block that can optionally change the number of channels. 可以自己选择通道数的数量
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        semb_channels,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        # Here we use the built component of SPADE, rather than SFT. Should have no significant influence on the performance.
        self.spade = SPADE(self.out_channels, semb_channels)

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb, s_cond):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb, s_cond), self.parameters(), self.use_checkpoint
        )


    def _forward(self, x, emb, s_cond):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        h = self.spade(h, s_cond)  #h=(6,1280,4,4)
        return self.skip_connection(x) + h

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)   # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        #return pt_checkpoint(self._forward, x)  # pytorch

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])

class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.attention_op: Optional[Any] = None

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        if XFORMERS_IS_AVAILBLE:
            q, k, v = map(
                lambda t:t.permute(0,2,1)
                .contiguous(),
                (q, k, v),
            )
            # actually compute the attention, what we cannot get enough of
            a = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)
            a = (
                a.permute(0,2,1)
                .reshape(bs, -1, length)
            )
        else:
            weight = th.einsum(
                "bct,bcs->bts", q * scale, k * scale
            )  # More stable with f16 than dividing afterwards
            weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
            a = th.einsum("bts,bcs->bct", weight, v)
            a = a.reshape(bs, -1, length)
        return a

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.attention_op: Optional[Any] = None

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        if XFORMERS_IS_AVAILBLE:
            q, k, v = map(
                lambda t:t.permute(0,2,1)
                .contiguous(),
                (q, k, v),
            )
            # actually compute the attention, what we cannot get enough of
            a = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)
            a = (
                a.permute(0,2,1)
                .reshape(bs, -1, length)
            )
        else:
            weight = th.einsum(
                "bct,bcs->bts",
                (q * scale).view(bs * self.n_heads, ch, length),
                (k * scale).view(bs * self.n_heads, ch, length),
            )  # More stable with f16 than dividing afterwards
            weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
            a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
            a = a.reshape(bs, -1, length)
        return a

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support 
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'
        # in_channels=6#这里先采取concat的形式
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            normalization(ch),
            conv_nd(dims, model_channels, n_embed, 1),
            #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps=None, context=None, y=None,**kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)

class UNetModelDualcondV2(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        semb_channels=None
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'
    #use_STF 和context_dim必须都指明清楚
        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),  #x<0: y=0, x>0:y=x
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels #320
        input_block_chans = [model_channels] #[320]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):  #(index,channel_mult[index])
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformerV2(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlockDual(
                ch,
                time_embed_dim,
                dropout,
                semb_channels=semb_channels,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformerV2(  # always uses a self-attn
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                            disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                            use_checkpoint=use_checkpoint
                        ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlockDual(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        semb_channels=semb_channels,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformerV2(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlockDual(
                            ch,
                            time_embed_dim,
                            dropout,
                            semb_channels=semb_channels,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            normalization(ch),
            conv_nd(dims, model_channels, n_embed, 1),
            #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps=None, context=None, moe_prompt_cond=None, y=None,**kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        #这里出现的问题：
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h) # Concat with decoder
        h = self.middle_block(h, emb, context,moe_prompt_cond)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context, moe_prompt_cond)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)