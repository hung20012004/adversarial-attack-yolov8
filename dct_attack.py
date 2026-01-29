"""
DCT-based Adversarial Attack for Object Detection
YOLOv8 Compatible Version
"""

import torch
import numpy as np
import time
import random
from tqdm import tqdm
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.metrics import box_iou


def set_seed(seed=0):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def xywh2xyxy(x):
    """Convert [x, y, w, h] to [x1, y1, x2, y2]"""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """Rescale boxes from img1_shape to img0_shape"""
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    boxes[..., [0, 2]] -= pad[0]
    boxes[..., [1, 3]] -= pad[1]
    boxes[..., :4] /= gain
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, img0_shape[1])
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, img0_shape[0])
    return boxes


def compute_ap(recall, precision):
    """Compute AP"""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    try:
        ap = np.trapezoid(np.interp(x, mrec, mpre), x)
    except AttributeError:
        ap = np.trapz(np.interp(x, mrec, mpre), x)
    return ap, mpre, mrec


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir=None, names=(), eps=1e-16):
    """Compute AP per class"""
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]
    px, py = np.linspace(0, 1, 1000), []
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l, n_p = nt[ci], i.sum()
        if n_p == 0 or n_l == 0:
            continue
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)
        recall = tpc / (n_l + eps)
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)
        precision = tpc / (tpc + fpc)
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
    
    f1 = 2 * p * r / (p + r + eps)
    i = f1.mean(0).argmax()
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt.reshape((-1, 1))).round()
    return tp, fpc, p, r, f1, ap, unique_classes.astype(int)


def create_frequency_mask(H, W, freq_type, r_h=0.2, r_w=0.2, d_min=0.2, d_max=0.6, device='cpu'):
    """Create frequency mask"""
    mask = torch.zeros((1, 1, H, W), device=device, dtype=torch.float32)
    
    if freq_type == 'low':
        size_h, size_w = int(r_h * H), int(r_w * W)
        mask[:, :, :size_h, :size_w] = 1
    elif freq_type == 'high':
        size_h, size_w = int(r_h * H), int(r_w * W)
        mask[:, :, :, :] = 1
        mask[:, :, :size_h, :size_w] = 0
    elif freq_type == 'mid':
        i_coords = torch.arange(H, device='cpu').float().unsqueeze(1) / H
        j_coords = torch.arange(W, device='cpu').float().unsqueeze(0) / W
        dist = torch.sqrt(i_coords**2 + j_coords**2)
        mid_mask = (dist > d_min) & (dist < d_max)
        mask[0, 0] = mid_mask.to(device)
    elif freq_type == 'all':
        mask[:, :, :, :] = 1
    else:
        raise ValueError(f"freq_type must be 'low', 'mid', 'high', or 'all'")
    
    return mask


_DCT_MATRIX_CACHE = {}


class DCT4OD_mAP:
    """DCT-based Adversarial Attack for YOLOv8"""
    
    def __init__(self, model, epsilon, max_iter, step_size, target_map_reduction,
                 conf_thres, iou_thres, img_size, names=None, freq_type='low',
                 r_h=0.2, r_w=0.2, d_min=0.2, d_max=0.6):
        self.model = model
        self.epsilon = epsilon if epsilon <= 1.0 else epsilon / 255.0
        self.max_iter = max_iter
        self.step_size = step_size if step_size <= 1.0 else step_size / 255.0
        self.target_map_reduction = target_map_reduction
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.names = names if names is not None else {}
        self.img_size = img_size
        self.device = next(model.parameters()).device
        self.dtype = torch.float32
        
        print(f"\n{'='*80}")
        print(f"Initializing Full-DCT Attack (YOLOv8)")
        print(f"  Image Size: {img_size}×{img_size}")
        print(f"  Frequency: {freq_type}")
        print(f"  Epsilon: {epsilon}")
        print(f"  Step Size: {step_size}")
        print(f"  Max Iterations: {max_iter}")
        print(f"{'='*80}\n")
        
        self.dct_matrix = self._create_dct_matrix(img_size).to(self.device, dtype=self.dtype)
        self.freq_mask = create_frequency_mask(
            H=img_size, W=img_size, freq_type=freq_type,
            r_h=r_h, r_w=r_w, d_min=d_min, d_max=d_max,
            device=self.device
        )
        
        active_ratio = (self.freq_mask.sum() / (img_size * img_size)) * 100
        print(f"Active frequency components: {active_ratio:.1f}%\n")

    def _extract_predictions(self, model_output):
        """
        Extract predictions tensor from model output
        YOLOv8 can return: tuple, dict, or tensor
        """
        if isinstance(model_output, tuple):
            return model_output[0]
        elif isinstance(model_output, dict):
            for key in ['pred', 'predictions', 'output', 'logits']:
                if key in model_output and isinstance(model_output[key], torch.Tensor):
                    return model_output[key]
            for v in model_output.values():
                if isinstance(v, torch.Tensor):
                    return v
            raise ValueError(f"Cannot extract tensor from dict with keys: {model_output.keys()}")
        else:
            return model_output

    def _create_dct_matrix(self, n):
        """Create DCT matrix with caching"""
        cache_key = (n, str(self.device), str(self.dtype))
        
        if cache_key in _DCT_MATRIX_CACHE:
            print(f"✓ Using cached DCT matrix ({n}×{n})")
            return _DCT_MATRIX_CACHE[cache_key]
        
        print(f"Creating DCT matrix ({n}×{n})... ", end='', flush=True)
        start_time = time.time()
        
        k = torch.arange(n, dtype=self.dtype, device='cpu').view(-1, 1)
        i = torch.arange(n, dtype=self.dtype, device='cpu').view(1, -1)
        
        dct_m = torch.where(
            k == 0,
            torch.sqrt(torch.tensor(1.0 / n, dtype=self.dtype)),
            torch.sqrt(torch.tensor(2.0 / n, dtype=self.dtype)) * 
            torch.cos(torch.pi * k * (2 * i + 1) / (2 * n))
        )
        
        _DCT_MATRIX_CACHE[cache_key] = dct_m
        print(f"Done! ({time.time() - start_time:.2f}s)")
        return dct_m

    def full_dct_2d(self, x):
        """2D DCT transform"""
        B, C, H, W = x.shape
        x_2d = x.view(-1, H, W)
        x_dct = torch.matmul(self.dct_matrix, x_2d)
        x_dct = torch.matmul(x_dct, self.dct_matrix.T)
        return x_dct.view(B, C, H, W)

    def full_idct_2d(self, x_dct):
        """2D Inverse DCT transform"""
        B, C, H, W = x_dct.shape
        x_dct_2d = x_dct.view(-1, H, W)
        x = torch.matmul(self.dct_matrix.T, x_dct_2d)
        x = torch.matmul(x, self.dct_matrix)
        return x.view(B, C, H, W)

    def compute_map(self, predictions, targets, img_tensor, shapes):
        """Compute mAP for YOLOv8 predictions"""
        try:
            if isinstance(predictions, tuple):
                predictions = predictions[0]  
            elif isinstance(predictions, dict):
                for key in ['pred', 'predictions', 'output']:
                    if key in predictions and isinstance(predictions[key], torch.Tensor):
                        predictions = predictions[key]
                        break
            
            predictions = predictions.detach()

            output = non_max_suppression(
                predictions.clone(), 
                self.conf_thres, 
                self.iou_thres, 
                multi_label=True
            )
            
            iouv = torch.linspace(0.5, 0.95, 10).to(self.device)
            stats = []
            
            for si, pred in enumerate(output):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []
                
                if len(pred) == 0:
                    if nl: 
                        stats.append((
                            torch.zeros(0, 10, dtype=torch.bool), 
                            torch.Tensor(), 
                            torch.Tensor(), 
                            tcls
                        ))
                    continue
                
                correct = torch.zeros(pred.shape[0], 10, dtype=torch.bool, device=self.device)
                
                if nl:
                    tcls_tensor = labels[:, 0]
                    tbox = xywh2xyxy(labels[:, 1:5])
                    scale_boxes(img_tensor[si].shape[1:], tbox, shapes[si][0], shapes[si][1])
                    predn = pred.clone()
                    scale_boxes(img_tensor[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])
                    
                    detected = []
                    for cls in torch.unique(tcls_tensor):
                        ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)
                        pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)
                        
                        if pi.shape[0]:
                            ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)
                            for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                d = ti[i[j]]
                                if d.item() not in detected:
                                    detected.append(d.item())
                                    correct[pi[j]] = ious[j] > iouv
                                    if len(detected) == nl: 
                                        break
                
                stats.append((
                    correct.cpu().detach(), 
                    pred[:, 4].cpu().detach(), 
                    pred[:, 5].cpu().detach(), 
                    tcls
                ))
            
            if len(stats) and stats[0][0].any():
                stats_np = [np.concatenate(x, 0) for x in zip(*stats)]
                tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats_np, plot=False, names=self.names)
                return {'map': float(ap.mean()), 'map50': float(ap[:, 0].mean())}
            
            return {'map': 0.0, 'map50': 0.0}
        
        except Exception as e:
            print(f"Warning in compute_map: {e}")
            return {'map': 0.0, 'map50': 0.0}

    def run(self, original_image, targets, shapes, image_path, save_dir):
        """Run adversarial attack"""
        start_time = time.time()
        nb, _, height, width = original_image.shape
        
        original_model_dtype = next(self.model.parameters()).dtype
        
        self.model.float()
        for param in self.model.parameters():
            param.data = param.data.float()
        for buffer in self.model.buffers():
            buffer.data = buffer.data.float()
        
        original_image = original_image.detach().float()
        
        targets_scaled = targets.clone().detach()
        if targets_scaled.shape[1] >= 6:
            targets_scaled[:, 2:6] *= torch.Tensor([width, height, width, height]).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            model_output = self.model(original_image)
            orig_preds = self._extract_predictions(model_output)
            metrics_orig = self.compute_map(orig_preds, targets_scaled, original_image, shapes)
            orig_map = metrics_orig['map']
        
        target_map = orig_map * (1 - self.target_map_reduction)
        print(f"Original mAP: {orig_map:.4f} | Target: {target_map:.4f}")
        
        adversarial_image = original_image.clone().detach().requires_grad_(True)
        best_map = orig_map
        best_adv = original_image.clone().detach()
        check_interval = max(1, self.max_iter // 50)
        
        pbar = tqdm(range(self.max_iter), desc="Attacking", ncols=100)
        
        for i in pbar:
            self.model.eval()
            
            model_output = self.model(adversarial_image)
            
            try:
                preds = self._extract_predictions(model_output)
            except Exception as e:
                print(f"\n⚠ Error extracting predictions at iter {i}: {e}")
                break
            
            if preds.dim() >= 3:
                if preds.shape[1] > preds.shape[2]:
                    preds = preds.transpose(1, 2).transpose(2, 3)
                    loss = preds[..., 4].sum() 
                else:
                    loss = preds[..., 4].sum() if preds.shape[-1] > 4 else preds.max()
            else:
                loss = preds.sum()
            
            if not loss.requires_grad:
                print(f"\nLoss has no gradient at iter {i}")
                break
            
            if loss.item() == 0:
                print(f"\nZero loss at iter {i}, stopping")
                break
            
            loss.backward()

            if adversarial_image.grad is None:
                print(f"\nNo gradient at iter {i}")
                break

            with torch.no_grad():
                grad = adversarial_image.grad.clone()
                grad_dct = self.full_dct_2d(grad)
                grad_dct_masked = grad_dct * self.freq_mask
                noise = self.full_idct_2d(grad_dct_masked)
                
                norm = torch.linalg.norm(noise) + 1e-10
                adversarial_image.data -= self.step_size * (noise / norm)
                
                pert = (adversarial_image.data - original_image.data).clamp(-self.epsilon, self.epsilon)
                adversarial_image.data = (original_image.data + pert).clamp(0, 1)

            adversarial_image.grad.zero_()
            self.model.zero_grad()
            
            if i % 10 == 0:
                torch.cuda.empty_cache()

            if i % check_interval == 0 or i == self.max_iter - 1:
                self.model.eval()
                with torch.no_grad():
                    eval_output = self.model(adversarial_image.detach())
                    curr_preds = self._extract_predictions(eval_output)
                    
                    curr_metrics = self.compute_map(
                        curr_preds, targets_scaled, 
                        adversarial_image.detach(), shapes
                    )
                    curr_map = curr_metrics['map']
                    
                    if curr_map < best_map:
                        best_map = curr_map
                        best_adv = adversarial_image.clone().detach()
                    
                    pbar.set_postfix({
                        'mAP': f'{curr_map:.4f}',
                        'best': f'{best_map:.4f}',
                        'target': f'{target_map:.4f}'
                    })
                    
                    if curr_map <= target_map:
                        print(f"\n✓ Target reached at iter {i+1}")
                        break

        pbar.close()
        
        attack_time = time.time() - start_time
        reduction = ((orig_map - best_map) / (orig_map + 1e-9)) * 100
        
        del adversarial_image
        if original_model_dtype == torch.float16:
            self.model.half()
            for param in self.model.parameters():
                param.data = param.data.half()
            for buffer in self.model.buffers():
                buffer.data = buffer.data.half()
        
        torch.cuda.empty_cache()
        
        success = best_map <= target_map
        
        print(f"\n{'='*80}")
        print(f"Attack {'SUCCESS' if success else 'FAILED'}")
        print(f"  Original mAP: {orig_map:.4f}")
        print(f"  Final mAP: {best_map:.4f}")
        print(f"  Reduction: {reduction:.1f}%")
        print(f"  Time: {attack_time:.1f}s")
        print(f"  Iterations: {i + 1}/{self.max_iter}")
        print(f"{'='*80}\n")

        return {
            'success': success,
            'adversarial_image': best_adv,
            'original_metrics': metrics_orig,
            'adversarial_metrics': {'map': best_map, 'map50': 0.0},
            'map_reduction_percent': reduction,
            'iterations_ran': i + 1,
            'attack_time': attack_time,
            'checkpoint_results': []
        }