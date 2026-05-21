import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torchvision.models as models
import clip
from typing import Literal

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        
        self.conv1 = nn.Conv2d(
            in_channels=in_channel, out_channels=out_channel,
            kernel_size=3, padding=1, stride=stride, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channel)
        
        self.conv2 = nn.Conv2d(
            in_channels=out_channel, out_channels=out_channel, 
            kernel_size=3, padding=1, stride=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channel)
        
        self.relu = nn.ReLU()
        
    def forward(self, X):
        output = self.relu(self.bn1(self.conv1(X)))
        output = self.bn2(self.conv2(output))
        
        if self.downsample is not None:
            shortcut = self.downsample(X)
        else:
            shortcut = X
        
        output += shortcut
        
        return self.relu(output)
        

class BottleNeckBlock(nn.Module):
    expansion = 4
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BottleNeckBlock, self).__init__()
        self.downsample = downsample
        
        self.conv1 = nn.Conv2d(
            in_channels=in_channel, out_channels=out_channel, kernel_size=1,
            stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channel)
        
        self.conv2 = nn.Conv2d(
            in_channels=out_channel, out_channels=out_channel, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channel)
        
        self.conv3 = nn.Conv2d(
            in_channels=out_channel, out_channels=out_channel * self.expansion, kernel_size=1,
            stride=1, padding=0, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        
        self.relu = nn.ReLU()
    
    def forward(self, X):
        output = self.relu(self.bn1(self.conv1(X)))
        output = self.relu(self.bn2(self.conv2(output)))
        output = self.bn3(self.conv3(output))
        
        if self.downsample is not None:
            shortcut = self.downsample(X)
        else:
            shortcut = X
        
        output += shortcut
        
        return self.relu(output)

class ResNet(nn.Module):
    def __init__(
        self, block,
        n_layer: Literal['50', '34', '18'] = '50' 
        ):
        
        super(ResNet, self).__init__()
        self.in_channel = 64
        
        if n_layer == '50' or n_layer == '34':
            layers = (3, 4, 6, 3)
        elif n_layer == '18':
            layers = (2, 2, 2, 2)
        
        self.n_layer = n_layer
        
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=self.in_channel, kernel_size=7,
            padding=3, stride=2, bias=False 
        )
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1
        )
        
        self.layer1 = self.create_layers(block, layers[0], 64)
        self.layer2 = self.create_layers(block, layers[1], 128, 2)
        self.layer3 = self.create_layers(block, layers[2], 256, 2)
        self.layer4 = self.create_layers(block, layers[3], 512, 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
    def create_layers(self, block, n_blocks, out_channel, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != out_channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channel, out_channels=out_channel * block.expansion,
                    kernel_size=1, padding=0, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channel * block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_channel, out_channel, stride, downsample))
        self.in_channel = out_channel * block.expansion
        
        for _ in range(1, n_blocks):
            layers.append(block(self.in_channel, out_channel))
            
        return nn.Sequential(*layers)
        
    def load_pretrained_resnet(self):
        if self.n_layer == '50':
            pretrained_resnet = models.resnet50(weights='DEFAULT')
        elif self.n_layer == '34':
            pretrained_resnet = models.resnet34(weights='DEFAULT')
        elif self.n_layer == '18':
            pretrained_resnet = models.resnet18(weights='DEFAULT')
            
        self.load_state_dict(pretrained_resnet.state_dict(), strict=False)    
        
        
        
    def forward(self, X):
        output = self.relu(self.bn1(self.conv1(X)))
        output = self.maxpool(output)
        
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        
        output = self.avgpool(output)
        output = torch.flatten(output, 1)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads=8, d_model=512, eps=1e9):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.eps = eps
        
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        d_model = Q.size(-1)
        d_k = d_model // self.n_heads
        
        Q_proj = self.Wq(Q).view(batch_size, Q.size(1), self.n_heads, d_k).transpose(1, 2)
        K_proj = self.Wk(K).view(batch_size, K.size(1), self.n_heads, d_k).transpose(1, 2)
        V_proj = self.Wv(V).view(batch_size, V.size(1), self.n_heads, d_k).transpose(1, 2)
        
        #(batch_size, n_heads, Q.size(1), K.size(1))
        scores = Q_proj @ torch.transpose(K_proj, -1, -2)
        scores = scores / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == False, -self.eps)
        
        attention_weights = torch.softmax(scores.float(), dim=-1).to(Q_proj.dtype)
        
        
        #(batch_size, n_heads, Q.size(1), d_k)
        output = attention_weights @ V_proj
        
        output = torch.transpose(output, 1, 2).contiguous().view(
            output.size(0), -1, d_model
        )
        
        output = self.Wo(output)
        
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super(FeedForward, self).__init__()
        d_ff = d_model * 4
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        
    def forward(self, X):
        output = self.linear1(X)
        output = nn.ReLU()(output)
        output = self.linear2(output)
        
        return output
        

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=76):
        super(PositionalEncoding, self).__init__()
        self.position_embeddings = nn.Parameter(torch.randn(max_len, d_model))
    
    def forward(self, X):
        return X + self.position_embeddings[:X.size(1), :]
        
class TransformerBlock(nn.Module):
    def __init__(self, n_heads, d_model, dropout):
        super(TransformerBlock, self).__init__()
        
        self.MHA = MultiHeadAttention(n_heads, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ff = FeedForward(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X, mask=None):
        nX = self.norm1(X)
        output = self.dropout(self.MHA(nX, nX, nX, mask))
        
        X = X + output
        
        output = self.dropout(self.ff(self.norm2(X)))
        
        return X + output
        
        
class Transformer(nn.Module):
    def __init__(self, d_model, max_len, n_heads, dropout, n_layers, vocab_size):
        super(Transformer, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.pe = PositionalEncoding(d_model, max_len)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(n_heads, d_model, dropout) for _  in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    @staticmethod
    def create_masks(X, pad_idx, device):
        X_mask = (X != pad_idx).unsqueeze(1).unsqueeze(2)
        
        #lower-trianglar 
        X_len = X.size(1)
        lt_mask = torch.tril(torch.ones((X_len, X_len), device=device)).bool()
        lt_mask = lt_mask.unsqueeze(0).unsqueeze(0)
        
        return X_mask & lt_mask
    
    def load_pretrained_transformer(self):
        pretrained_clip, _ = clip.load("ViT-B/32", device="cpu")
        checkpoint = pretrained_clip.transformer.state_dict()
        
        new_dict = {}

        new_dict['embedding.weight'] = pretrained_clip.token_embedding.weight
        new_dict['pe.position_embeddings'] = pretrained_clip.positional_embedding 

        for i in range(len(self.blocks)):
            src_prefix = f"resblocks.{i}."
            tgt_prefix = f"blocks.{i}."
            
            in_proj_weight = checkpoint[src_prefix + "attn.in_proj_weight"]
            in_proj_bias = checkpoint[src_prefix + "attn.in_proj_bias"]
            
            q_w, k_w, v_w = in_proj_weight.chunk(3, dim=0)
            q_b, k_b, v_b = in_proj_bias.chunk(3, dim=0)
            
            new_dict[tgt_prefix + "MHA.Wq.weight"] = q_w
            new_dict[tgt_prefix + "MHA.Wq.bias"] = q_b
            new_dict[tgt_prefix + "MHA.Wk.weight"] = k_w
            new_dict[tgt_prefix + "MHA.Wk.bias"] = k_b
            new_dict[tgt_prefix + "MHA.Wv.weight"] = v_w
            new_dict[tgt_prefix + "MHA.Wv.bias"] = v_b
            
            new_dict[tgt_prefix + "MHA.Wo.weight"] = checkpoint[src_prefix + "attn.out_proj.weight"]
            new_dict[tgt_prefix + "MHA.Wo.bias"] = checkpoint[src_prefix + "attn.out_proj.bias"]
            
            new_dict[tgt_prefix + "norm1.weight"] = checkpoint[src_prefix + "ln_1.weight"]
            new_dict[tgt_prefix + "norm1.bias"] = checkpoint[src_prefix + "ln_1.bias"]
            new_dict[tgt_prefix + "norm2.weight"] = checkpoint[src_prefix + "ln_2.weight"]
            new_dict[tgt_prefix + "norm2.bias"] = checkpoint[src_prefix + "ln_2.bias"]
            
            new_dict[tgt_prefix + "ff.linear1.weight"] = checkpoint[src_prefix + "mlp.c_fc.weight"]
            new_dict[tgt_prefix + "ff.linear1.bias"] = checkpoint[src_prefix + "mlp.c_fc.bias"]
            new_dict[tgt_prefix + "ff.linear2.weight"] = checkpoint[src_prefix + "mlp.c_proj.weight"]
            new_dict[tgt_prefix + "ff.linear2.bias"] = checkpoint[src_prefix + "mlp.c_proj.bias"]

        new_dict['norm.weight'] = pretrained_clip.ln_final.weight
        new_dict['norm.bias'] = pretrained_clip.ln_final.bias

        self.load_state_dict(new_dict, strict=False)
        
    def forward(self, X, eos_idx, pad_idx):
        mask = self.create_masks(X, pad_idx, X.device)
        
        X = self.embedding(X)
        X = self.pe(X)
        
        for block in self.blocks:
            X = block(X, mask)
        
        X = self.norm(X)
        
        seq_len = X.size(1)
        safe_eos_idx = torch.clamp(eos_idx, min=0, max=seq_len - 1)
        
        return X[torch.arange(X.size(0)), safe_eos_idx]
    
class CLIPScratch(nn.Module):
    def __init__(
        self, config,
        n_layer: Literal['50', '34', '18'] = '50' 
        ):
        super(CLIPScratch, self).__init__()
        
        self.img_encoder = ResNet(config.block, n_layer)
        self.text_encoder = Transformer(
            config.d_model, config.max_len, config.n_heads, config.dropout, config.n_layers, config.vocab_size
            )
        
        if n_layer == '50':
            img_enc_out_dim = 2048
        elif n_layer == '34' or n_layer == '18':
            img_enc_out_dim = 512
        
        self.Wi = nn.Linear(img_enc_out_dim, config.d_e)
        self.Wt = nn.Linear(config.text_enc_out_dim, config.d_e)
        
        self.t = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.apply(self.init_weights)
        nn.init.xavier_uniform_(self.Wi.weight)
        nn.init.xavier_uniform_(self.Wt.weight)
        nn.init.constant_(self.Wi.bias, 0)
        nn.init.constant_(self.Wt.bias, 0)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        
    def load_pretrained_all(self):
        self.img_encoder.load_pretrained_resnet() 
        
        self.text_encoder.load_pretrained_transformer()
        
    def forward(self, X, eos_idx, pad_idx):
        X_img, X_text = X
        
        X_img = self.img_encoder(X_img)
        X_text = self.text_encoder(X_text, eos_idx, pad_idx)
        
        X_img = self.Wi(X_img)
        X_text = self.Wt(X_text)
        
        X_img = F.normalize(X_img, p=2, dim=-1)
        X_text = F.normalize(X_text, p=2, dim=-1)
        
        t = self.t.clamp(max=4.6052)
        
        return (X_img @ X_text.t()) * torch.exp(t)
    