import logging
import math
import time
import torch
import numpy as np
from einops import rearrange

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_cast_dtype
from .distributed import is_master
from .zero_shot import zero_shot_eval
from .precision import get_autocast

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }

def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model

def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()

def no_op(*args, **kwargs):
    pass

def sample_gaussian_bounded(min_val=1, max_val=5):
        """
        Sample an integer between min_val and max_val using a Gaussian distribution centered at `mean`.
        
        Parameters:
        - min_val (int): Minimum value of the sampled integer (inclusive).
        - max_val (int): Maximum value of the sampled integer (inclusive).
        - std (float): Standard deviation of the Gaussian distribution.
        
        Returns:
        - int: Randomly sampled integer.
        """
        mean = (max_val + min_val) / 2  # Center around the middle value (3.5 for [1, 6])
        sampled_value = torch.normal(mean=float(mean), std=max(1.0,max_val/5), size=(1,)).item() 
        sampled_value = max(min_val, min(max_val, round(sampled_value)))
        return int(sampled_value)

def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, custom_logger, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        images, texts = batch
        texts = texts.to(device=device, non_blocking=True)

        if not args.video:
            images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        else:
            if args.distill_anchor_target:
                anchor_images = rearrange(images[:,0,:,:,:].unsqueeze(1), 'b f c h w -> (b f) c h w') 
                dist_images   = rearrange(images[:,1:,:,:,:],   'b f c h w -> (b f) c h w') 
                target_images = rearrange(images[:,1:,:,:,:], 'b f c h w -> (b f) c h w')  

                # No action needed if both are None
                if args.force_image_size is not None and args.force_dist_image_size is not None:
                    if args.force_image_size == args.force_dist_image_size:
                        # Skip resizing if sizes are the same
                        pass
                    elif args.force_image_size < args.force_dist_image_size:
                        dist_images = torch.nn.functional.interpolate(
                                dist_images, size=(args.force_image_size, args.force_image_size), 
                                mode='bilinear', align_corners=False
                            )
                    else:
                        raise ValueError("force_image_size should be less than or equal to force_dist_image_size")

                elif args.force_image_size is not None:
                    if anchor_images.shape[-2] != args.force_image_size:
                        dist_images = torch.nn.functional.interpolate(
                                dist_images, size=(args.force_image_size, args.force_image_size), 
                                mode='bilinear', align_corners=False
                            )

                elif args.force_dist_image_size is not None:
                    if anchor_images.shape[-2] != args.force_dist_image_size:
                        raise ValueError("force_dist_image_size should match the anchor image size")

                anchor_images = anchor_images.to(device=device, dtype=cast_dtype, non_blocking=True)
                dist_images   = dist_images.to(device=device, dtype=cast_dtype, non_blocking=True)
                target_images = target_images.to(device=device, dtype=cast_dtype, non_blocking=True) 

            else:
                images = rearrange(images, 'b f c h w -> (b f) c h w')
                images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
            
        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                model_out = {}
                if args.distill:
                    with torch.no_grad():
                        if args.distill_anchor_target:
                            dist_model_out = dist_model(target_images, texts)
                            anchor_model_out = dist_model(anchor_images, texts)
                        else:
                            dist_model_out = dist_model(images, texts)

                    model_out.update({f'dist_{k}' : v for k, v in dist_model_out.items()})
                    
                if args.distill_anchor_target:
                    num_frames = dist_images.shape[0] // anchor_images.shape[0]
                    residual_feat = anchor_model_out['image_features'].repeat_interleave(num_frames, dim=0)
                    tmp_out = model(dist_images, texts, residual_feat=residual_feat)
                else:
                    tmp_out = model(images, texts)
                
                model_out.update({k : v for k, v in tmp_out.items()})
                logit_scale = model_out["logit_scale"]
                
                losses = loss(**model_out, output_dict=True)

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)
                    model_out.pop("logit_scale")
                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)
                    logit_scale = model_out.pop("logit_scale")
                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] +  [model_out[key]] + accumulated[j + 1:])
                    losses = loss(**inputs, logit_scale=logit_scale, output_dict=True)
                    del inputs
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss
                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }            
            log_data.update({name:val.val for name,val in losses_m.items()})

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step}, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for
    return step

def evaluate(model, data, epoch, step, args, dist_model=None, loss=None, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    
    model.eval()

    metrics = {}
    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
    metrics.update(zero_shot_metrics)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        cumulative_loss = 0.0
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                if args.video:
                    B, F, C, H, W = images.shape
                    if args.distill_anchor_target:
                        anchor_images = images[:,0,:,:,:].reshape(-1, 1, C, H, W)
                        anchor_images = anchor_images.reshape(-1, C, H, W)
                        dist_images   = images[:,1:,:,:,:].reshape(-1, C, H, W)
                        target_images = images[:,1:,:,:,:].reshape(-1, C, H, W)

                        # No action needed if both are None
                        if args.force_image_size is not None and args.force_dist_image_size is not None:
                            if args.force_image_size == args.force_dist_image_size:
                                # Skip resizing if sizes are the same
                                pass
                            elif args.force_image_size < args.force_dist_image_size:
                                dist_images = torch.nn.functional.interpolate(
                                        dist_images, size=(args.force_image_size, args.force_image_size), 
                                        mode='bilinear', align_corners=False
                                    )
                            else:
                                raise ValueError("force_image_size should be less than or equal to force_dist_image_size")

                        elif args.force_image_size is not None:
                            if anchor_images.shape[-2] != args.force_image_size:
                                dist_images = torch.nn.functional.interpolate(
                                        dist_images, size=(args.force_image_size, args.force_image_size), 
                                        mode='bilinear', align_corners=False
                                    )

                        elif args.force_dist_image_size is not None:
                            if anchor_images.shape[-2] != args.force_dist_image_size:
                                raise ValueError("force_dist_image_size should match the anchor image size")

                    else:
                        images = images.view(-1, C, H, W)

                if args.distill_anchor_target:
                    anchor_images = anchor_images.to(device=device, dtype=cast_dtype, non_blocking=True)
                    dist_images   = dist_images.to(device=device, dtype=cast_dtype, non_blocking=True)
                    target_images = target_images.to(device=device, dtype=cast_dtype, non_blocking=True)
                else:
                    images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    model_out = {}
                    if args.distill:
                        with torch.no_grad():
                            if args.distill_anchor_target:
                                dist_model_out = dist_model(target_images, texts)
                                anchor_model_out = dist_model(anchor_images, texts)
                            else:
                                dist_model_out = dist_model(images, texts)
                        model_out.update({f'dist_{k}' : v for k, v in dist_model_out.items()})

                    if args.distill_anchor_target:
                        num_frames = dist_images.shape[0] // anchor_images.shape[0]
                        residual_feat = anchor_model_out['image_features'].repeat_interleave(num_frames, dim=0)
                        tmp_out = model(dist_images, texts, residual_feat=residual_feat)
                    else:
                        tmp_out = model(images, texts)

                    model_out.update({k : v for k, v in tmp_out.items()})

                    losses = loss(**model_out, output_dict=True)
                    total_loss = sum(losses.values())

                cumulative_loss += total_loss * B
                num_samples += B
            
            loss = cumulative_loss / num_samples
            metrics.update(
                {"val_loss": loss.item(), 
                "epoch": int(epoch), 
                "num_samples": int(num_samples)}
            )
                    

    if is_master(args):
        logging.info(
            f"Eval Epoch: {epoch} "
            + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
        )

        if args.wandb:
            assert wandb is not None, 'Please install wandb.'
            for name, val in metrics.items():
                wandb.log({f"val/{name}": val, 'epoch': epoch}, step=step)

def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics

def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)
