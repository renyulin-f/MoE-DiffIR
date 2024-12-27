import torch
import torch.nn as nn
from CLIP import clip
from CLIP.clip.model import ModifiedResNet
import open_clip

class Prompt_semantic(nn.Module):
    def __init__(self,prompt_len,prompt_channel,prompt_size) -> None:
        super().__init__()
        self.prompt_param=nn.Parameter(torch.rand(prompt_len,prompt_channel,prompt_size)-0.5,requires_grad=True)
        self.prompt_param=torch.nn.init.zeros_(self.prompt_param)
        
    def forward(self,x):
        p = self.prompt_param
        prompt = p
        batch,_,_=x.shape
        for i in range(batch-1):
            prompt=torch.cat((prompt,p),dim=0) 
        x = torch.cat((x,prompt),dim=1)
        return x

class CLIP_Semantic_extractor2(ModifiedResNet):
    def __init__(self, layers=(3, 4, 6, 3), pretrained=True,freeze=True,Type='semantic',path=None):
        super(CLIP_Semantic_extractor2, self).__init__(layers=layers, output_dim=1024, heads=32)
        self.Type = Type
        ckpt = 'RN50' if path is None else path
        if pretrained:
            model, _ = clip.load(ckpt, device='cpu')
        self.load_state_dict(model.visual.state_dict())
        self.requires_grad_(False)
        self.prompt_model_semantic = Prompt_semantic(1,49,1024)
        self.adapter1 = Adpater(1024)
        self.conv = nn.Sequential(
                nn.Conv2d(2048,1024,3,1,1),)
        for param in self.prompt_model_semantic.parameters():
            param.requires_grad = True #(256,1024)
        for param in self.adapter1.parameters():
            param.requires_grad = True #(256,1024)
        for param in self.conv.parameters():
            param.requires_grad = True #(256,1024)
        del model
    
    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x
        # b c h w
        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) #(B,2048,7,7)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.prompt_model_semantic(x)
        x = self.adapter1(x)
        # CLIP image encoder
        return x #(3,1024)

class Adpater(nn.Module):
    def __init__(self,c_in,reduction=4):
        super(Adpater,self).__init__()
        self.fc=nn.Sequential(nn.Linear(c_in,c_in//reduction,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in//reduction,c_in,bias=False),
            nn.ReLU(inplace=True))
    def forward(self,x):
        x_new = self.fc(x)
        return x + x_new
    
class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class FrozenOpenCLIPEmbedder(AbstractEncoder): #这里是fixed的openclip transformer encoder对于text
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        #"pooled",
        "last",
        "penultimate"
    ]
    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device=torch.device('cuda'), max_length=77,
                 freeze=True, layer="last",Type='text'):
        super().__init__()
        assert layer in self.LAYERS  #这里是设置decive的重点：
        model, _, _ = open_clip.create_model_and_transforms(arch, device=device, pretrained=version)
        del model.visual
        self.model = model
        self.Type = Type
        self.device = device
        self.max_length = max_length
        if freeze: #固定住参数，paramters.requires_grad=False
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x #(3,7,1024)

    def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)
