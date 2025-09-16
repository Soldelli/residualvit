from collections import OrderedDict
import math
from typing import Callable, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .custom_multihead_attention import CustomMultiheadAttention

from .utils import to_2tuple


class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, force_inference_patch_dropout=False, 
                inference_patch_dropout_mode='uniform',
                exclude_first_token=True, patch_size=16):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token
        self.force_inference_patch_dropout = force_inference_patch_dropout
        self.subsampler = None
        self.patch_size = patch_size
        if self.force_inference_patch_dropout:
            if inference_patch_dropout_mode == 'random': 
                self.subsampler = self.random_subsampler
            elif inference_patch_dropout_mode == 'uniform':
                self.subsampler = self.uniform_subsampler
            elif inference_patch_dropout_mode == 'center': 
                self.subsampler = self.centercrop_subsampler
            elif inference_patch_dropout_mode == 'motion': 
                self.subsampler = self.motion_based_subsampler
            elif inference_patch_dropout_mode == 'residuals': 
                self.subsampler = self.residuals_based_subsampler

    def uniform_subsampler(self, x_input, N):
        """
        Random subsample N elements along dimension 1 from a 3D PyTorch tensor.
        
        Parameters:
        - x (torch.Tensor): The input 3D tensor to subsample from, shape [B, P, C].
        - N (int): The number of elements to subsample along dimension 1.
        
        Returns:
        - torch.Tensor: A tensor containing N randomly subsampled elements, shape [B, N, C].
        """
        B, P, C = x_input.shape
        indices = torch.linspace(0, P-1, steps=N, dtype=torch.long)
        return x_input[:, indices]

    def random_subsampler(self, x_input, N):
        """
        Uniformly subsample N elements along dimension 1 from a 3D PyTorch tensor.
        
        Parameters:
        - x (torch.Tensor): The input 3D tensor to subsample from, shape [B, P, C].
        - N (int): The number of elements to subsample along dimension 1.
        
        Returns:
        - torch.Tensor: A tensor containing N uniformly subsampled elements, shape [B, N, C].
        """
        B, P, C = x_input.shape

        rand = torch.randn(B, P)
        patch_indices_keep = rand.topk(N, dim=-1).indices
        batch_indices = torch.arange(B)[..., None]

        return x_input[batch_indices, patch_indices_keep]

    def centercrop_subsampler(self, x_input, N):
        """
        Uniformly subsample N patches in a concentric square manner from a 3D PyTorch tensor.
        
        Parameters:
        - x (torch.Tensor): The input 3D tensor to subsample from, shape [B, P, C].
        - N (int): The number of patches to subsample.
        
        Returns:
        - torch.Tensor: A tensor containing N concentric square subsampled patches, shape [B, N, C].
        """
        
        B, P, C = x_input.shape
        
        # Compute grid size from total patches
        grid_size = int(P ** 0.5)
        
        # Adjusted center calculation
        center = grid_size / 2.0 - 0.5
        
        # Compute distance from the center for each patch
        y, x = torch.div(torch.arange(grid_size * grid_size), grid_size), torch.remainder(torch.arange(grid_size * grid_size), grid_size)
        distances = (x - center) ** 2 + (y - center) ** 2
        
        # Sort by distances, keeping track of the original indices
        sorted_indices = distances.argsort()
        
        # Extract the first N indices for subsampling
        indices_to_keep = sorted_indices[:N]
        
        # Extract the first N indices for subsampling
        ordered_indices = indices_to_keep.sort().values
        
        return x_input[:, ordered_indices] #, ordered_indices

    def motion_based_subsampler(self, x_input, N, motion_vectors):
        """
        Uniformly subsample N patches in a concentric square manner from a 3D PyTorch tensor.
        
        Parameters:
        - x (torch.Tensor): The input 3D tensor to subsample from, shape [B, P, C].
        - N (int): The number of patches to subsample.
        
        Returns:
        - torch.Tensor: A tensor containing N concentric square subsampled patches, shape [B, N, C].
        """
        
        B = x_input.shape[0]

        # Flatten video into images batch
        if len(motion_vectors.shape)==4:
            motion_vectors = motion_vectors.reshape(-1, *motion_vectors.shape[2:])

        # Compute motion per patch
        blocks = motion_vectors.unfold(1, self.patch_size, self.patch_size)
        blocks = blocks.unfold(2, self.patch_size, self.patch_size)
        block_sums = blocks.reshape(B, -1, self.patch_size*self.patch_size).sum(dim=2)

        #  Select the top N patches with the most motion, add a little bit of noise to mimic random sampling for patches with zero motion
        block_sums += torch.randn(block_sums.shape, device=block_sums.device, dtype=torch.float32) * 0.000001
        patch_indices_keep = block_sums.topk(N, dim=1).indices
        
        batch_indices = torch.arange(B).unsqueeze(1)
        return x_input[batch_indices, patch_indices_keep] #, ordered_indices
    
    def residuals_based_subsampler(self, x_input, N, residuals):
        """
        Uniformly subsample N patches in a concentric square manner from a 3D PyTorch tensor.
        
        Parameters:
        - x (torch.Tensor): The input 3D tensor to subsample from, shape [B, P, C].
        - N (int): The number of patches to subsample.
        
        Returns:
        - torch.Tensor: A tensor containing N concentric square subsampled patches, shape [B, N, C].
        """
        
        B = x_input.shape[0]

        # Flatten video into images batch
        if len(residuals.shape)==4:
            residuals = residuals.view(-1, *residuals.shape[2:])

        # Compute residuals per patch
        blocks = residuals.unfold(1, self.patch_size, self.patch_size)
        blocks = blocks.unfold(2, self.patch_size, self.patch_size)
        block_sums = blocks.reshape(B, -1, self.patch_size*self.patch_size).sum(dim=2)
        patch_indices_keep = block_sums.topk(N, dim=1).indices
        
        batch_indices = torch.arange(B).unsqueeze(1)
        return x_input[batch_indices, patch_indices_keep] #, ordered_indices

    def forward(self, x, motion_vectors=None, residuals=None):
        if self.prob == 0.:
            return x

        if self.training:
            if self.exclude_first_token:
                cls_tokens, x = x[:, :1], x[:, 1:]
            else:
                cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

            batch = x.size()[0]
            num_tokens = x.size()[1]

            batch_indices = torch.arange(batch)
            batch_indices = batch_indices[..., None]

            keep_prob = 1 - self.prob
            num_patches_keep = max(1, int(num_tokens * keep_prob))

            rand = torch.randn(batch, num_tokens)
            patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

            x = x[batch_indices, patch_indices_keep]

            if self.exclude_first_token:
                x = torch.cat((cls_tokens, x), dim=1)

        elif not self.training and self.force_inference_patch_dropout:
            if self.exclude_first_token:
                cls_tokens, x = x[:, :1], x[:, 1:]
            else:
                cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

            keep_prob = 1 - self.prob
            num_tokens = x.size()[1]
            # num_patches_keep = max(1, int(round(num_tokens * keep_prob)))
            num_patches_keep = int(max(1, torch.round(torch.tensor(num_tokens * keep_prob))).item())

            params = {
                'x_input': x,
                'N': num_patches_keep,
            }

            if motion_vectors is not None:
                params['motion_vectors'] = motion_vectors

            if residuals is not None:
                params['residuals'] = residuals

            x = self.subsampler(**params)

            if self.exclude_first_token:
                x = torch.cat((cls_tokens, x), dim=1)

        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            scaled_cosine=False,
            scale_heads=False,
            logit_scale_max=math.log(1. / 0.01),
            attn_drop=0.,
            proj_drop=0.
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale_max = logit_scale_max

        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        self.in_proj_weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale)
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
        else:
            self.in_proj_bias = None

        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        L, N, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        k = k.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        v = v.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)

        if self.logit_scale is not None:
            attn = torch.bmm(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2))
            logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn.view(N, self.num_heads, L, L) * logit_scale
            attn = attn.view(-1, L, L)
        else:
            q = q * self.scale
            attn = torch.bmm(q, k.transpose(-1, -2))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            attn += attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.bmm(attn, v)
        if self.head_scale is not None:
            x = x.view(N, self.num_heads, L, C) * self.head_scale
            x = x.view(-1, L, C)
        x = x.transpose(0, 1).reshape(L, N, C)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x 


class AttentionalPooler(nn.Module):
    def __init__(
            self,
            d_model: int,
            context_dim: int,
            n_head: int = 8,
            n_queries: int = 256,
            norm_layer: Callable = LayerNorm
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_head, kdim=context_dim, vdim=context_dim)
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def forward(self, x: torch.Tensor):
        x = self.ln_k(x).permute(1, 0, 2)  # NLD -> LND
        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(self._repeat(q, N), x, x, need_weights=False)[0]
        return out.permute(1, 0, 2)  # LND -> NLD

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)


class PatchMerging(nn.Module):
    '''
    Adapted from https://arxiv.org/pdf/2210.09461.pdf 
    '''
    def __init__(self, r: int=0, class_token: bool = True, use_residual_token: bool = False):
        super().__init__()
        self.r = r
        self.class_token = class_token
        self.residual_token = use_residual_token

    def bipartite_soft_matching(self, metric: torch.Tensor) -> Callable:
        """
        Applies ToMe with a balanced matching set (50%, 50%).

        Input size is [batch, tokens, channels].
        r indicates the number of tokens to remove (max 50% of tokens).

        Extra args:
        - class_token: Whether or not there's a class token.
        - residual_token: Whether or not there's also a residual token token.

        When enabled, the class token and residual_token tokens won't get merged.
        """

        # Define useful functions
        def do_nothing(x, mode=None):
            return x

        def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
            src, dst = x[..., ::2, :], x[..., 1::2, :]
            n, t1, c = src.shape
            unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
            src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
            dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

            if self.residual_token:
                return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
            else:
                return torch.cat([unm, dst], dim=1)

        # Apply algorithm
        protected = 0
        if self.class_token:
            protected += 1
        if self.residual_token:
            protected += 1

        # We can only reduce by a maximum of 50% tokens
        t = metric.shape[1]
        r = min(self.r, (t - protected) // 2)

        if r <= 0:
            return do_nothing

        with torch.no_grad():
            metric = metric / metric.norm(dim=-1, keepdim=True)
            a, b = metric[..., ::2, :], metric[..., 1::2, :]
            scores = a @ b.transpose(-1, -2)

            if self.class_token:
                scores[..., 0, :] = -math.inf
            if self.residual_token:
                scores[..., :, 0] = -math.inf

            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

            unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
            src_idx = edge_idx[..., :r, :]  # Merged Tokens
            dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

            if self.class_token:
                # Sort to ensure the class token is at the start
                unm_idx = unm_idx.sort(dim=1)[0]

        return merge

    def merge_wavg(self, merge: Callable, x: torch.Tensor, size: torch.Tensor = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the merge function by taking a weighted average based on token size.
        Returns the merged tensor and the new token sizes.
        """
        if size is None:
            size = torch.ones_like(x[..., 0, None])

        x = merge(x * size, mode="sum")
        size = merge(size, mode="sum")
        return x / size, size

    def forward(self, x: torch.Tensor, metric: torch.Tensor, size: torch.Tensor = None):
        merge = self.bipartite_soft_matching(metric)
        x, size = self.merge_wavg(merge, x.transpose(1,0), size)
        return x.transpose(1,0), size


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            is_cross_attention: bool = False,
            use_residual_token: bool = False,
            use_patch_merging: bool = False,
            patch_merging_r: int = 0,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.use_patch_merging = use_patch_merging
        if self.use_patch_merging:
            self.attn = CustomMultiheadAttention(d_model, n_head)
            self.patch_merging = PatchMerging(r=patch_merging_r, 
                                use_residual_token=use_residual_token)
        else:
            self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            attn_size: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None

        if self.use_patch_merging:
            attn_outputs, _, k = self.attn(q_x, k_x, v_x, need_weights=True, 
                                    attn_mask=attn_mask, attn_size=attn_size)
            return attn_outputs, k.mean(1)
        else:
            return self.attn(q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask)[0], None

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            attn_size: Optional[torch.Tensor] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None

        xx, metric = self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, 
                                    attn_mask=attn_mask, attn_size=attn_size)
        x = q_x + self.ls_1(xx)

        # here goes the patch merging
        if self.use_patch_merging and metric is not None:
            x, attn_size = self.patch_merging(x, metric, attn_size)
        else:
            attn_size = None

        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x, attn_size


class CustomResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            scale_cosine_attn: bool = False,
            scale_heads: bool = False,
            scale_attn: bool = False,
            scale_fc: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = Attention(
            d_model, n_head,
            scaled_cosine=scale_cosine_attn,
            scale_heads=scale_heads,
        )
        self.ln_attn = norm_layer(d_model) if scale_attn else nn.Identity()
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ('ln', norm_layer(mlp_width) if scale_fc else nn.Identity()),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, 
                attn_size: Optional[torch.Tensor] = None):

        xx, attn_size = self.attn(self.ln_1(x), attn_mask=attn_mask, attn_size=attn_size)
        x = x + self.ls_1(self.ln_attn(xx))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x, attn_size


class Transformer(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            use_patch_merging: bool = False,
            use_residual_token: bool = False,
            patch_merging_r: int = 0,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
                ResidualAttentionBlock(
                        width,
                        heads, 
                        mlp_ratio, 
                        ls_init_value=ls_init_value, 
                        act_layer=act_layer, 
                        norm_layer=norm_layer,
                        use_patch_merging=use_patch_merging,
                        use_residual_token=use_residual_token,
                        patch_merging_r=patch_merging_r,
                    )
            for _ in range(layers)
        ])

    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):

        attn_size = None # type: Optional[torch.Tensor] This parameters is used by patch merging
        # We need to propagate it all the way out here as each residual block is initialized as an independent
        # Object and it is messy to pass things otherwise. 

        all_levels_cls_tokens = []
        for l, r in enumerate(self.resblocks):

            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                x, attn_size = r(x, attn_mask=attn_mask, attn_size=attn_size)

        return x, None


class VisionTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            ls_init_value: float = None,
            global_average_pool: bool = False,
            attentional_pool: bool = False,
            n_queries: int = 256,
            attn_pooler_heads: int = 8,
            output_dim: int = 512,
            patch_dropout: float = 0.,
            force_inference_patch_dropout: bool = False,
            inference_patch_dropout_mode: str = 'uniform',
            input_patchnorm: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False,
            use_residual_token: bool = False,
            residual_token_projection_num_layers: int = 1,
            residual_token_dim: int = 512,
            use_patch_merging: bool = False,
            patch_merging_r: int = 0,
    ):
        super().__init__()
        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.output_dim = output_dim

        # whether to layernorm each patch, as done in dual patchnorm paper - https://arxiv.org/abs/2302.01327v1
        self.input_patchnorm = input_patchnorm

        if input_patchnorm:
            patch_input_dim = patch_height * patch_width * 3
            self.patchnorm_pre_ln = LayerNorm(patch_input_dim)
            self.conv1 = nn.Linear(patch_input_dim, width)
        else:
            self.patchnorm_pre_ln = nn.Identity()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        # class embeddings and positional embeddings
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = nn.Identity()
        if patch_dropout >= 0.:
            self.patch_dropout = PatchDropout(
                                    patch_dropout, 
                                    force_inference_patch_dropout,
                                    inference_patch_dropout_mode,
                                    patch_size=patch_size
                                )

        self.ln_pre = norm_layer(width)
        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            use_patch_merging=use_patch_merging,
            use_residual_token=use_residual_token,
            patch_merging_r=patch_merging_r,
        )

        self.global_average_pool = global_average_pool
        if attentional_pool:
            self.attn_pool = AttentionalPooler(output_dim, width, n_head=attn_pooler_heads, n_queries=n_queries)
            self.ln_post = norm_layer(output_dim)
            self.proj = nn.Parameter(scale * torch.randn(output_dim, output_dim))
        else:
            self.attn_pool = None
            self.ln_post = norm_layer(width)
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.use_residual_token = use_residual_token
        if self.use_residual_token:

            layers = [nn.Linear(residual_token_dim, width)]
            
            for i in range(residual_token_projection_num_layers-1):
                layers += [nn.ReLU(), nn.Linear(width, width)]

            self.projection_residual_feature = nn.Sequential(*layers)            
            self.proj_head = nn.Sequential(*layers)

        self.init_parameters()

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        for param in self.parameters():
            param.requires_grad = False

        if unlocked_groups != 0:
            groups = [
                [
                    self.conv1,
                    self.class_embedding,
                    self.positional_embedding,
                    self.ln_pre,
                ],
                *self.transformer.resblocks[:-1],
                [
                    self.transformer.resblocks[-1],
                    self.ln_post,
                ],
                self.proj,
            ]

            def _unlock(x):
                if isinstance(x, Sequence):
                    for g in x:
                        _unlock(g)
                else:
                    if isinstance(x, torch.nn.Parameter):
                        x.requires_grad = True
                    else:
                        for p in x.parameters():
                            p.requires_grad = True

            _unlock(groups[-unlocked_groups:])

        if self.use_residual_token:
            for p in self.projection_residual_feature.parameters():
                p.requires_grad = True

    def init_parameters(self):
        # FIXME OpenAI CLIP did not define an init for the VisualTransformer
        # TODO experiment if default PyTorch init, below, or alternate init is best.

        # nn.init.normal_(self.class_embedding, std=self.scale)
        # nn.init.normal_(self.positional_embedding, std=self.scale)
        #
        # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        # attn_std = self.transformer.width ** -0.5
        # fc_std = (2 * self.transformer.width) ** -0.5
        # for block in self.transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        #
        # if self.text_projection is not None:
        #     nn.init.normal_(self.text_projection, std=self.scale)
        pass

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.global_average_pool:
            return x.mean(dim=1), x
        else:
            return x[:, 0], x[:, 1:]

    def forward(self, x: torch.Tensor, motion_vectors=None, residuals=None, residual_feat=None):
        # to patches - whether to use dual patchnorm - https://arxiv.org/abs/2302.01327v1
        if self.input_patchnorm:
            # einops - rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
            x = x.reshape(x.shape[0], x.shape[1], self.grid_size[0], self.patch_size[0], self.grid_size[1], self.patch_size[1])
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(x.shape[0], self.grid_size[0] * self.grid_size[1], -1)
            x = self.patchnorm_pre_ln(x)
            x = self.conv1(x)
        else:
            x = self.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.patch_dropout(x, motion_vectors, residuals)

        if self.use_residual_token:
            assert residual_feat is not None
            residual_token = self.projection_residual_feature(residual_feat.unsqueeze(1))
            x = torch.cat([x[:,0].unsqueeze(1), residual_token, x[:,1:]], dim=1)
            
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, cls_tokens = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.attn_pool is not None:
            x = self.attn_pool(x)
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)
        else:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)

        if self.proj is not None:
            pooled = pooled @ self.proj

        output_dict = {
            'features': pooled
        }
        
        if self.output_tokens:
            output_dict['tokens'] = tokens
        
        return output_dict


class TextTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            context_length: int = 77,
            vocab_size: int = 49408,
            width: int = 512,
            heads: int = 8,
            layers: int = 12,
            ls_init_value: float = None,
            output_dim: int = 512,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            embed_cls: bool = False,
            pad_id: int = 0,
            output_tokens: bool = False,
    ):
        super().__init__()
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pad_id = pad_id

        self.text_projection = nn.Parameter(torch.empty(width, output_dim))

        if embed_cls:
            self.cls_emb = nn.Parameter(torch.empty(width))
            self.num_pos += 1
        else:
            self.cls_emb = None

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(self.num_pos, width))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer
        )
        self.ln_final = norm_layer(width)

        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if self.cls_emb is not None:
            nn.init.normal_(self.cls_emb, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def build_cls_mask(self, text, cast_dtype: torch.dtype):
        cls_mask = (text != self.pad_id).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=1.0)
        additive_mask = torch.empty(cls_mask.shape, dtype=cast_dtype, device=cls_mask.device)
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        additive_mask = torch.repeat_interleave(additive_mask, self.heads, 0)
        return additive_mask

    def _repeat(self, t, N: int):
        return t.reshape(1, 1, -1).repeat(N, 1, 1)

    def forward(self, text):
        cast_dtype = self.transformer.get_cast_dtype()
        seq_len = text.shape[1]

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        attn_mask = self.attn_mask
        if self.cls_emb is not None:
            seq_len += 1
            x = torch.cat([x, self._repeat(self.cls_emb, x.shape[0])], dim=1)
            cls_mask = self.build_cls_mask(text, cast_dtype)
            attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=attn_mask)[0]
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if self.cls_emb is not None:
            pooled, tokens = x[:, -1], x[:, :-1]
            pooled = self.ln_final(pooled)
        else:
            x = self.ln_final(x)
            pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x

        if self.text_projection is not None:
            pooled = pooled @ self.text_projection

        if self.output_tokens:
            return pooled, tokens

        return pooled


class MultimodalTransformer(Transformer):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            context_length: int = 77,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_dim: int = 512,
    ):

        super().__init__(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.context_length = context_length
        self.cross_attn = nn.ModuleList([
            ResidualAttentionBlock(
                width,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                is_cross_attention=True,
            )
            for _ in range(layers)
        ])

        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

        self.ln_final = norm_layer(width)
        self.text_projection = nn.Parameter(torch.empty(width, output_dim))

    def init_parameters(self):
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        for block in self.transformer.cross_attn:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, image_embs, text_embs):
        text_embs = text_embs.permute(1, 0, 2)  # NLD -> LNDsq
        image_embs = image_embs.permute(1, 0, 2)  # NLD -> LND
        seq_len = text_embs.shape[0]

        for resblock, cross_attn in zip(self.resblocks, self.cross_attn):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                text_embs = checkpoint(resblock, text_embs, None, None, self.attn_mask[:seq_len, :seq_len])
                text_embs = checkpoint(cross_attn, text_embs, image_embs, image_embs, None)
            else:
                text_embs = resblock(text_embs, attn_mask=self.attn_mask[:seq_len, :seq_len])
                text_embs = cross_attn(text_embs, k_x=image_embs, v_x=image_embs)

        x = text_embs.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        if self.text_projection is not None:
            x = x @ self.text_projection

        return x

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable
