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
            alpha=1.0,
            nl_semantic_supervision=False,
            semantic_weight=1.0,
            semantic_pairwise=True
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
        self.alpha = alpha

        # Natural Language Semantic Supervision
        self.nl_semantic_supervision = nl_semantic_supervision
        self.semantic_pairwise = semantic_pairwise

        if nl_semantic_supervision:
            self.semantic_weight = semantic_weight

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

    def forward(self, image_features, text_features, logit_scale, output_dict=False, semantic_features=None):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        clip_loss = self.alpha*((
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        )/2)
        if self.nl_semantic_supervision:
            if self.semantic_pairwise:
                semantic_loss = 0
                for i in range(text_features.shape[0]):
                    in_clip_distance = text_features - text_features[i]
                    in_semantic_distance = semantic_features - semantic_features[i]
                    intermediate_semantic_loss = F.mse_loss(in_clip_distance, in_semantic_distance)
                    semantic_loss = semantic_loss + intermediate_semantic_loss
                semantic_loss = semantic_loss / text_features.shape[0]
                semantic_loss = self.semantic_weight * semantic_loss
            else:
                semantic_loss = self.semantic_weight*(F.mse_loss(text_features, semantic_features))
            
            total_loss = clip_loss + semantic_loss
            return {"total_loss": total_loss, "clip_loss": clip_loss, "semantic_loss": semantic_loss} if output_dict else total_loss
        else:
            total_loss = clip_loss
            return {"contrastive_loss": total_loss} if output_dict else total_loss


class ClipInModalityLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            adaptive=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            alpha=1.0,
            beta=0.5,
            n_epoch=30,
            epoch=1,
            nl_semantic_supervision=False,
            semantic_weight=1.0
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
        if adaptive:
            self.alpha = epoch/n_epoch
            self.beta = 1 - self.alpha
        else:
            self.alpha = alpha
            self.beta = beta
        
        self.nl_semantic_supervision = nl_semantic_supervision

        if nl_semantic_supervision:
            self.semantic_weight = semantic_weight

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
                logits_per_image = image_features @ all_image_features.T
                logits_per_text = text_features @ all_text_features.T

                logits_image_text = image_features @ text_features.T
                size = logits_per_image.shape[0]

                logscale_logits_image_text = logit_scale * image_features @ all_text_features.T
                logscale_logits_text_image = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = all_image_features @ all_image_features.T
                logits_per_text = all_text_features @ all_text_features.T

                logits_image_text = all_image_features @ all_text_features.T
                size = logits_per_image.shape[0]
                
                logscale_logits_image_text = logit_scale * all_image_features @ all_text_features.T
                logscale_logits_text_image = logscale_logits_image_text.T 
        else:
            logits_per_image = image_features @ image_features.T
            logits_per_text = text_features @ text_features.T
            
            logits_image_text = image_features @ text_features.T
            size = logits_per_image.shape[0]
            
            logscale_logits_image_text = logit_scale * image_features @ text_features.T
            logscale_logits_text_image = logit_scale * text_features @ image_features.T
        
        device = logits_per_image.get_device()
        identity_complement = 1 - torch.eye(size).to(device)

        paired_logits_image_text = torch.mul(torch.eye(size).to(device), logits_image_text)                
        paired_logits_image_text = identity_complement + paired_logits_image_text

        logits_per_image = logit_scale * torch.mul(logits_per_image, paired_logits_image_text)

        logits_per_text = logit_scale * torch.mul(logits_per_text, paired_logits_image_text)
        
        return logits_per_image, logits_per_text, logscale_logits_image_text, logscale_logits_text_image

    def forward(self, image_features, text_features, logit_scale, output_dict=False, semantic_features=None):
        device = image_features.device
        logits_per_image, logits_per_text, logscale_logits_image_text, logscale_logits_text_image = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        if self.nl_semantic_supervision:

            semantic_features = semantic_features / semantic_features.norm(dim=-1, keepdim=True)
            semantic_sim = semantic_features @ semantic_features.T
            semantic_sim = 1 - semantic_sim

            size = logits_per_image.shape[0]

            logits_per_text = torch.mul(logits_per_text, semantic_sim)
            device = logits_per_image.get_device()

            logits_paired_text_image = torch.mul(logits_per_image, torch.eye(size).to(device))

            logits_per_text = logits_per_text + logits_paired_text_image

            inModality_loss = self.beta*((F.cross_entropy(logits_per_text, labels)))
        else:
            inModality_loss = self.beta*((
                F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_text, labels)
                )/2)
        
        clip_loss = self.alpha*((
            F.cross_entropy(logscale_logits_image_text, labels) +
            F.cross_entropy(logscale_logits_text_image, labels)
        )/2)

        total_loss = inModality_loss + clip_loss
        return {"total_loss": total_loss, "clip_loss": clip_loss, "inModality_loss": inModality_loss} if output_dict else total_loss


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
        
        clip_loss = torch.tensor(0)
        
        if self.clip_loss_weight:
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

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss
