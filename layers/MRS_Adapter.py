import math
import torch
import torch.nn as nn

import clip

class MRS_Adapter(nn.Module):
    def __init__(self,
                 d_model=512,
                 bottleneck=64,
                 dropout=0.5
                 ):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck
        self.d = 64

        self.visual_down_proj = nn.Linear(768, self.down_size)
        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()

        self.up_proj = nn.Linear(self.down_size, self.n_embd-self.d)
        self.visual_up_proj = nn.Linear(self.down_size, 768-self.d)

        self.share_up_proj = nn.Linear(self.down_size, self.d)

        self.dropout = dropout

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.visual_down_proj.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.visual_up_proj.weight)
            nn.init.zeros_(self.share_up_proj.weight)
            nn.init.zeros_(self.visual_down_proj.bias)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)
            nn.init.zeros_(self.visual_up_proj.bias)
            nn.init.zeros_(self.share_up_proj.bias)

    def forward(self, x, mode='text'):
        if mode == 'text':
            down = self.down_proj(x.float())
        elif mode == 'visual':
            down = self.visual_down_proj(x.float())

        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)

        if mode == 'text':
            up_unique = self.up_proj(down)
        elif mode == 'visual':
            up_unique = self.visual_up_proj(down)

        up_share = self.share_up_proj(down)
        output = torch.cat((up_unique,up_share), dim=2)

        return output



class PE_RSITR(nn.Module):
    def __init__(self, patch_size=32, ):
        super(PE_RSITR, self).__init__()

        assert patch_size == 14 or patch_size == 16 or patch_size == 32
        self.patch_size = patch_size
        if self.patch_size == 14:
            self.model, self.preprocess = clip.load("ViT-L/{}".format(self.patch_size))
        else:
            self.model, self.preprocess = clip.load("ViT-B/{}".format(self.patch_size))

        self.Uniadapt = MRS_Adapter()
        self.dropout = nn.Dropout(0.2)

        self.dtype = self.model.dtype

    def encode_text(self, text):
        return self.model.encode_text(text)

    def encode_image(self, image):
        return self.model.encode_image(image)


    def forward_text(self, text, augmentation = False):
        trans = self.model.transformer
        x = self.model.token_embedding(text).type(self.dtype)

        x = x + self.model.positional_embedding.type(self.dtype)
        if augmentation:
            x = self.dropout(x)
        x = x.permute(1, 0, 2)

        for block_idx, resblock in enumerate(trans.resblocks):
            attn = resblock.attention
            x = x + attn(resblock.ln_1(x))

            adapt_x = self.Uniadapt(x, 'text').to(x.dtype)
            residual = x
            x = resblock.mlp(resblock.ln_2(x))
            x = x + adapt_x
            x = residual + x

        x = x.permute(1, 0, 2)
        x = self.model.ln_final(x).type(self.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.model.text_projection  # [bs, 512]

        return x


    def forward_image(self, im, augmentation = False):
        vit = self.model.visual
        x = im.type(self.model.dtype)

        x = vit.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([vit.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                     dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + vit.positional_embedding.to(x.dtype)

        x = vit.ln_pre(x)

        if augmentation:
            x = self.dropout(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        for block_idx, resblock in enumerate(vit.transformer.resblocks):
            attn = resblock.attention
            x = x + attn(resblock.ln_1(x))

            adapt_x = self.Uniadapt(x, 'visual').to(x.dtype)
            residual = x
            x = resblock.mlp(resblock.ln_2(x))

            x = x + adapt_x
            x = residual + x

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = vit.ln_post(x[:, 0, :])

        if vit.proj is not None:
            x = x @ vit.proj

        return x


