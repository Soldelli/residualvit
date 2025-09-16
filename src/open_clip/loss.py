import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):
        clip_loss = super().forward(image_features, text_features, logit_scale)
        clip_loss = self.clip_loss_weight * clip_loss

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss


class DistillClipLoss(ClipLoss):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            no_contrastive_loss=False,
            distill_vision_tower_only=False,
            distill_anchor_target=False,
            num_frames=1, 
            loss_version='ce', #'mse'
        ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )
        self.no_contrastive_loss = no_contrastive_loss
        self.distill_vision_tower_only = distill_vision_tower_only
        self.distill_anchor_target = distill_anchor_target
        self.num_frames = num_frames
        self.pool_frame_dimension = False
    
        #  Define which function works as forward
        self.forward = getattr(self, f'loss_{loss_version}')

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def _info_nce_loss(self, x, y, logit_scale):
        x_to_y, y_to_x = self.get_logits(x, y, logit_scale)
        labels = self.get_ground_truth(x.device, x_to_y.shape[0])
        return (F.cross_entropy(x_to_y, labels) + F.cross_entropy(y_to_x, labels)) / 2

    def loss_ce(self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
            ):

        '''
            This loss is the same of loss v1 but we removed the residual summation. 
            The idea is to use this loss to distill CLIP into the smaller model for 
            figuring out the best distillation possible. 
        '''

        feat_dims = image_features.shape[-1]
        num_frames = self.num_frames -1 if self.distill_anchor_target else self.num_frames

        if int(torch.norm(image_features, dim=-1).sum()) != image_features.shape[0]:
                image_features = F.normalize(image_features, dim=-1)

        if int(torch.norm(dist_image_features, dim=-1).sum()) != dist_image_features.shape[0]:
            dist_image_features = F.normalize(dist_image_features, dim=-1)

        if int(torch.norm(text_features, dim=-1).sum()) != text_features.shape[0]:
                text_features = F.normalize(text_features, dim=-1)

        if int(torch.norm(dist_text_features, dim=-1).sum()) != dist_text_features.shape[0]:
                dist_text_features = F.normalize(dist_text_features, dim=-1)

        image_features = image_features.view(-1, num_frames, feat_dims)
        dist_image_features = dist_image_features.view(-1, num_frames, feat_dims)

        distill_loss = []
        for i in range(num_frames):
            image_features_ = image_features[:,i].contiguous()
            dist_image_features_ = dist_image_features[:,i].contiguous()

            logits_per_image, logits_per_text = \
                self.get_logits(image_features_, text_features, logit_scale)

            dist_logits_per_image, dist_logits_per_text = \
                self.get_logits(dist_image_features_, dist_text_features, dist_logit_scale)

            if self.no_contrastive_loss:
                contrastive_loss = torch.tensor(0.)
            else:
                labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

                contrastive_loss = (
                    F.cross_entropy(logits_per_image, labels) +
                    F.cross_entropy(logits_per_text, labels)
                ) / 2

            if self.distill_vision_tower_only:
                distill_loss.append(self.dist_loss(dist_logits_per_image, logits_per_image))
            else:
                distill_loss.append((
                    self.dist_loss(dist_logits_per_image, logits_per_image) +
                    self.dist_loss(dist_logits_per_text, logits_per_text)
                ) / 2)
        
        distill_loss = torch.stack(distill_loss).mean()

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss

    def loss_mse(self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,

            ):

        '''
            This loss is the same of loss v1 but we removed the residual summation. 
            The idea is to use this loss to distill CLIP into the smaller model for 
            figuring out the best distillation possible. 
        '''

        if int(torch.norm(image_features, dim=-1).sum()) != image_features.shape[0]:
                image_features = F.normalize(image_features, dim=-1)

        if int(torch.norm(dist_image_features, dim=-1).sum()) != dist_image_features.shape[0]:
            dist_image_features = F.normalize(dist_image_features, dim=-1)

        distill_loss = F.mse_loss(image_features, dist_image_features, reduction='none') 
        distill_loss = distill_loss.sum(-1) / np.sqrt(image_features.shape[-1])
        distill_loss = 100 * distill_loss.mean()
        contrastive_loss = torch.tensor(0.)

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss
    