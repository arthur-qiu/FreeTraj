from functools import partial
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False
from lvdm.common import (
    checkpoint,
    exists,
    default,
)
from lvdm.basics import (
    zero_module,
)

from utils.utils_freetraj import get_path, plan_path
import math

def gaussian_2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    """ 2d Gaussian weight function
    """
    gaussian_map = (
        1
        / (2 * math.pi * sx * sy)
        * torch.exp(-((x - mx) ** 2 / (2 * sx**2) + (y - my) ** 2 / (2 * sy**2)))
    )
    gaussian_map.div_(gaussian_map.max())
    return gaussian_map

def gaussian_weight(height=32, width=32, KERNEL_DIVISION=3.0):

    x = torch.linspace(0, height, height)
    y = torch.linspace(0, width, width)
    x, y = torch.meshgrid(x, y, indexing="ij")
    noise_patch = (
                    gaussian_2d(
                        x,
                        y,
                        mx=int(height / 2),
                        my=int(width / 2),
                        sx=float(height / KERNEL_DIVISION),
                        sy=float(width / KERNEL_DIVISION),
                    )
                ).half()
    return noise_patch

class RelativePosition(nn.Module):
    """ https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py """

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        device = self.embeddings_table.device
        range_vec_q = torch.arange(length_q, device=device)
        range_vec_k = torch.arange(length_k, device=device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = final_mat.long()
        embeddings = self.embeddings_table[final_mat]
        return embeddings


class CrossAttention(nn.Module):

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., 
                 relative_position=False, temporal_length=None, img_cross_attention=False):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

        self.image_cross_attention_scale = 1.0
        self.text_context_len = 77
        self.img_cross_attention = img_cross_attention
        if self.img_cross_attention:
            self.to_k_ip = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v_ip = nn.Linear(context_dim, inner_dim, bias=False)
        
        self.relative_position = relative_position
        if self.relative_position:
            assert(temporal_length is not None)
            self.relative_position_k = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)
            self.relative_position_v = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)
        else:
            ## only used for spatial attention, while NOT for temporal attention
            if XFORMERS_IS_AVAILBLE and temporal_length is None:
                self.forward = self.space_forward

    def forward(self, x, context=None, mask=None, use_freetraj=False, idx_list=[], input_traj=[]):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        ## considering image token additionally
        if context is not None and self.img_cross_attention:
            context, context_img = context[:,:self.text_context_len,:], context[:,self.text_context_len:,:]
            k = self.to_k(context)
            v = self.to_v(context)
            k_ip = self.to_k_ip(context_img)
            v_ip = self.to_v_ip(context_img)
        else:
            k = self.to_k(context)
            v = self.to_v(context)

        hw = q.shape[0]
        w_base = 64
        h_base = 40
        w_len = int((hw / w_base / h_base) ** 0.5 * h_base)
        h_len = int(hw / w_len)
        BOX_SIZE_H = input_traj[0][2] - input_traj[0][1]
        BOX_SIZE_W = input_traj[0][4] - input_traj[0][3]
        PATHS = plan_path(input_traj)
        sub_h = int(BOX_SIZE_H * h_len) 
        sub_w = int(BOX_SIZE_W * w_len)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        if self.relative_position:
            len_q, len_k, len_v = q.shape[1], k.shape[1], v.shape[1]
            k2 = self.relative_position_k(len_q, len_k)
            sim2 = einsum('b t d, t s d -> b t s', q, k2) * self.scale # TODO check 
            sim += sim2
        del k

        if use_freetraj:
            sim = rearrange(sim, '(y x h) i j -> y x h i j', h=h, y=h_len)
            sim_mask = torch.zeros_like(sim)
            for i in range(sim.shape[3]):
                h_start1 = int(PATHS[i][0] * h_len)
                h_end1 = h_start1 + sub_h
                w_start1 = int(PATHS[i][2] * w_len)
                w_end1 = w_start1 + sub_w

                h_fg1 = list(range(h_start1, h_end1))
                h_fg_tensor1 = torch.zeros(h_len, device=sim.device)
                h_fg_tensor1[h_fg1] = 1
                w_fg1 = list(range(w_start1, w_end1))
                w_fg_tensor1 = torch.zeros(w_len, device=sim.device)
                w_fg_tensor1[w_fg1] = 1
                fg_tensor1 = h_fg_tensor1.view(-1, 1) * w_fg_tensor1.view(1, -1)
                bg_tensor1 = 1 - fg_tensor1

                for j in range(sim.shape[4]):
                    h_start2 = int(PATHS[j][0] * h_len)
                    h_end2 = h_start2 + sub_h
                    w_start2 = int(PATHS[j][2] * w_len)
                    w_end2 = w_start2 + sub_w

                    h_fg2 = list(range(h_start2, h_end2))
                    h_fg_tensor2 = torch.zeros(h_len, device=sim.device)
                    h_fg_tensor2[h_fg2] = 1
                    w_fg2 = list(range(w_start2, w_end2))
                    w_fg_tensor2 = torch.zeros(w_len, device=sim.device)
                    w_fg_tensor2[w_fg2] = 1
                    fg_tensor2 = h_fg_tensor2.view(-1, 1) * w_fg_tensor2.view(1, -1)
                    bg_tensor2 = 1 - fg_tensor2
                    fg_tensor = fg_tensor1 * fg_tensor2
                    bg_tensor = bg_tensor1 * bg_tensor2

                    coef = 0.01
                    sim_mask[:, :, :, i, j] = coef * torch.ones_like(sim_mask[:, :, :, i, j])
                    sim_mask[:, :, :, i, j] += (1 - coef) * torch.ones_like(sim_mask[:, :, :, i, j]) * (fg_tensor.view(h_len, w_len, 1) + bg_tensor.view(h_len, w_len, 1))

            sim *= sim_mask
            sim = rearrange(sim, 'y x h i j -> (y x h) i j')

            del sim_mask

        if exists(mask):
            ## feasible for causal attention mask only
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b i j -> (b h) i j', h=h)
            sim.masked_fill_(~(mask>0.5), max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', sim, v)
        if self.relative_position:
            v2 = self.relative_position_v(len_q, len_v)
            out2 = einsum('b t s, t s d -> b t d', sim, v2) # TODO check
            out += out2
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        ## considering image token additionally
        if context is not None and self.img_cross_attention:
            k_ip, v_ip = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (k_ip, v_ip))
            sim_ip =  torch.einsum('b i d, b j d -> b i j', q, k_ip) * self.scale
            del k_ip
            sim_ip = sim_ip.softmax(dim=-1)
            out_ip = torch.einsum('b i j, b j d -> b i d', sim_ip, v_ip)
            out_ip = rearrange(out_ip, '(b h) n d -> b n (h d)', h=h)
            out = out + self.image_cross_attention_scale * out_ip
        del q

        return self.to_out(out)
    
    def space_forward(self, x, context=None, mask=None, use_freetraj=False, idx_list=[], input_traj=[]):
        
        if context is None:
            SA_flag = True
        else:
            SA_flag = False

        h = self.heads
        
        q = self.to_q(x)
        context = default(context, x)

        ## considering image token additionally
        if context is not None and self.img_cross_attention:
            context, context_img = context[:,:self.text_context_len,:], context[:,self.text_context_len:,:]
            k = self.to_k(context)
            v = self.to_v(context)
            k_ip = self.to_k_ip(context_img)
            v_ip = self.to_v_ip(context_img)
        else:
            k = self.to_k(context)
            v = self.to_v(context)

        hw = q.shape[1]
        w_base = 64
        h_base = 40
        w_len = int((hw / h_base / w_base) ** 0.5 * w_base)
        h_len = int(hw / w_len)
        BOX_SIZE_H = input_traj[0][2] - input_traj[0][1]
        BOX_SIZE_W = input_traj[0][4] - input_traj[0][3]
        PATHS = plan_path(input_traj)
        sub_h = int(BOX_SIZE_H * h_len) 
        sub_w = int(BOX_SIZE_W * w_len)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        if self.relative_position:
            len_q, len_k, len_v = q.shape[1], k.shape[1], v.shape[1]
            k2 = self.relative_position_k(len_q, len_k)
            sim2 = einsum('b t d, t s d -> b t s', q, k2) * self.scale # TODO check 
            sim += sim2
        del k

        if use_freetraj:
            coef_a = 0.25 / (BOX_SIZE_H * BOX_SIZE_W) / len(idx_list)
            weight = gaussian_weight(sub_h, sub_w).to(x.device)
            if SA_flag:
                weight_add = 0

                sim = rearrange(sim, '(t h) (y x) (y0 x0) -> t h y x y0 x0', h=h, y=h_len, y0=h_len)
                sim_mask = torch.zeros_like(sim)
                for i in range(sim.shape[0]):
                    h_start = int(PATHS[i][0] * h_len)
                    h_end = h_start + sub_h
                    w_start = int(PATHS[i][2] * w_len)
                    w_end = w_start + sub_w

                    h_fg = list(range(h_start, h_end))
                    h_fg_tensor = torch.zeros(h_len, device=sim.device)
                    h_fg_tensor[h_fg] = 1
                    w_fg = list(range(w_start, w_end))
                    w_fg_tensor = torch.zeros(w_len, device=sim.device)
                    w_fg_tensor[w_fg] = 1
                    fg_tensor = h_fg_tensor.view(-1, 1) * w_fg_tensor.view(1, -1)
                    bg_tensor = 1 - fg_tensor

                    coef = 0.01
                    sim_mask[i] = coef * torch.ones_like(sim_mask[i])
                    sim_mask[i] += (1-coef) * (torch.ones_like(sim_mask[i]) * fg_tensor.view(1, h_len, w_len, 1, 1) * fg_tensor.view(1, 1, 1, h_len, w_len) + torch.ones_like(sim_mask[i]) * bg_tensor.view(1, h_len, w_len, 1, 1) * bg_tensor.view(1, 1, 1, h_len, w_len))
                
                sim *= sim_mask
                sim = rearrange(sim, 't h y x y0 x0 -> (t h) (y x) (y0 x0)')    

            else:
                sim = rearrange(sim, '(t h) (y x) d -> t h y x d', h=h, y=h_len)
                sim_mask = torch.zeros_like(sim)
                weight_add = torch.zeros_like(sim)
                weight_map = torch.zeros([sim.shape[0], h_len, w_len], device=sim.device)
                for i in range(sim.shape[0]):
                    h_start = int(PATHS[i][0] * h_len)
                    h_end = h_start + sub_h
                    w_start = int(PATHS[i][2] * w_len)
                    w_end = w_start + sub_w

                    h_fg = list(range(h_start, h_end))
                    h_fg_tensor = torch.zeros(h_len, device=sim.device)
                    h_fg_tensor[h_fg] = 1
                    w_fg = list(range(w_start, w_end))
                    w_fg_tensor = torch.zeros(w_len, device=sim.device)
                    w_fg_tensor[w_fg] = 1
                    fg_tensor = h_fg_tensor.view(-1, 1) * w_fg_tensor.view(1, -1)
                    bg_tensor = 1 - fg_tensor
                    if idx_list == []:
                        p_fg = [2]
                    else:
                        p_fg = idx_list
                    p_bg = list(range(77))
                    for j in p_fg:
                        p_bg.remove(j)

                    weight_map[i, h_start:h_end, w_start:w_end] = weight * coef_a
                    sim_mask[i, :, :, :, p_bg] = torch.ones_like(sim_mask[i, :, :, :, p_bg]) * bg_tensor.view(1, h_len, w_len, 1)
                    weight_add[i, :, :, :, p_fg] = torch.ones_like(sim_mask[i, :, :, :, p_fg]) * weight_map[i].view(1, h_len, w_len, 1)

                max_neg_value = -torch.finfo(sim.dtype).max
                sim.masked_fill_(~(sim_mask>0.5), max_neg_value)
                sim = rearrange(sim, 't h y x d -> (t h) (y x) d')
                weight_add = rearrange(weight_add, 't h y x d -> (t h) (y x) d')

            del sim_mask

        if exists(mask):
            ## feasible for causal attention mask only
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b i j -> (b h) i j', h=h)
            sim.masked_fill_(~(mask>0.5), max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)
        if use_freetraj:
            sim += weight_add

        out = torch.einsum('b i j, b j d -> b i d', sim, v)
        if self.relative_position:
            v2 = self.relative_position_v(len_q, len_v)
            out2 = einsum('b t s, t s d -> b t d', sim, v2) # TODO check
            out += out2
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        ## considering image token additionally
        if context is not None and self.img_cross_attention:
            k_ip, v_ip = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (k_ip, v_ip))
            sim_ip =  torch.einsum('b i d, b j d -> b i j', q, k_ip) * self.scale
            del k_ip
            sim_ip = sim_ip.softmax(dim=-1)
            out_ip = torch.einsum('b i j, b j d -> b i d', sim_ip, v_ip)
            out_ip = rearrange(out_ip, '(b h) n d -> b n (h d)', h=h)
            out = out + self.image_cross_attention_scale * out_ip
        del q

        return self.to_out(out)


class BasicTransformerBlock(nn.Module):

    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                disable_self_attn=False, attention_cls=None, img_cross_attention=False):
        super().__init__()
        attn_cls = CrossAttention if attention_cls is None else attention_cls
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout,
            img_cross_attention=img_cross_attention)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, mask=None, use_freetraj=False, idx_list=[], input_traj=[], **kwargs):
        ## implementation tricks: because checkpointing doesn't support non-tensor (e.g. None or scalar) arguments
        input_tuple = (x,)      ## should not be (x), otherwise *input_tuple will decouple x into multiple arguments
        if context is not None:
            input_tuple = (x, context)
        if mask is not None:
            forward_mask = partial(self._forward, mask=mask)
            return checkpoint(forward_mask, (x,), self.parameters(), self.checkpoint)
        if context is not None and mask is not None:
            input_tuple = (x, context, mask)
        input_tuple = (x, context, mask, use_freetraj, idx_list, input_traj)
        return checkpoint(self._forward, input_tuple, self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, mask=None, use_freetraj=False, idx_list=[], input_traj=[]):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None, mask=mask, use_freetraj=use_freetraj, idx_list=idx_list, input_traj=input_traj) + x
        x = self.attn2(self.norm2(x), context=context, mask=mask, use_freetraj=use_freetraj, idx_list=idx_list, input_traj=input_traj) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data in spatial axis.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0., context_dim=None,
                 use_checkpoint=True, disable_self_attn=False, use_linear=False, img_cross_attention=False):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                n_heads,
                d_head,
                dropout=dropout,
                context_dim=context_dim,
                img_cross_attention=img_cross_attention,
                disable_self_attn=disable_self_attn,
                checkpoint=use_checkpoint) for d in range(depth)
        ])
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear


    def forward(self, x, context=None, **kwargs):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context, **kwargs)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in
    
    
class TemporalTransformer(nn.Module):
    """
    Transformer block for image-like data in temporal axis.
    First, reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0., context_dim=None,
                 use_checkpoint=True, use_linear=False, only_self_att=True, causal_attention=False,
                 relative_position=False, temporal_length=None):
        super().__init__()
        self.only_self_att = only_self_att
        self.relative_position = relative_position
        self.causal_attention = causal_attention
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Conv1d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        if not use_linear:
            self.proj_in = nn.Conv1d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        if relative_position:
            assert(temporal_length is not None)
            attention_cls = partial(CrossAttention, relative_position=True, temporal_length=temporal_length)
        else:
            attention_cls = partial(CrossAttention, temporal_length=temporal_length)
            # attention_cls = None
        if self.causal_attention:
            assert(temporal_length is not None)
            self.mask = torch.tril(torch.ones([1, temporal_length, temporal_length]))

        if self.only_self_att:
            context_dim = None
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                n_heads,
                d_head,
                dropout=dropout,
                context_dim=context_dim,
                attention_cls=attention_cls,
                checkpoint=use_checkpoint) for d in range(depth)
        ])
        if not use_linear:
            self.proj_out = zero_module(nn.Conv1d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

    def forward(self, x, context=None, **kwargs):
        b, c, t, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = rearrange(x, 'b c t h w -> (b h w) c t').contiguous()
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'bhw c t -> bhw t c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)

        if self.causal_attention:
            mask = self.mask.to(x.device)
            mask = repeat(mask, 'l i j -> (l bhw) i j', bhw=b*h*w)
        else:
            mask = None

        if self.only_self_att:
            ## note: if no context is given, cross-attention defaults to self-attention
            for i, block in enumerate(self.transformer_blocks):
                x = block(x, mask=mask, **kwargs)
            x = rearrange(x, '(b hw) t c -> b hw t c', b=b).contiguous()
        else:
            x = rearrange(x, '(b hw) t c -> b hw t c', b=b).contiguous()
            context = rearrange(context, '(b t) l con -> b t l con', t=t).contiguous()
            for i, block in enumerate(self.transformer_blocks):
                # calculate each batch one by one (since number in shape could not greater then 65,535 for some package)
                for j in range(b):
                    context_j = repeat(
                        context[j],
                        't l con -> (t r) l con', r=(h * w) // t, t=t).contiguous()
                    ## note: causal mask will not applied in cross-attention case
                    x[j] = block(x[j], context=context_j, **kwargs)
        
        if self.use_linear:
            x = self.proj_out(x)
            x = rearrange(x, 'b (h w) t c -> b c t h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = rearrange(x, 'b hw t c -> (b hw) c t').contiguous()
            x = self.proj_out(x)
            x = rearrange(x, '(b h w) c t -> b c t h w', b=b, h=h, w=w).contiguous()

        return x + x_in
    

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x, **kwargs):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_
