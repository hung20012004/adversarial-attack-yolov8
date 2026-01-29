import argparse
import csv
import time
import yaml
import cv2  
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.metrics import ConfusionMatrix, box_iou
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG

if Path('dct_attack.py').exists():
    from dct_attack import DCT4OD_mAP, set_seed
    set_seed(0)
else:
    print("WARNING: dct_attack.py not found")

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
    """Compute AP from recall and precision"""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    try:
        ap = np.trapezoid(np.interp(x, mrec, mpre), x)
    except AttributeError:
        ap = np.trapz(np.interp(x, mrec, mpre), x)
    return ap, mpre, mrec

def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir=Path(), names=(), eps=1e-16):
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

def colorstr(*args):
    colors = {'test': '\033[34m', 'Attack': '\033[31m'}
    string = ' '.join([str(x) for x in args])
    for key, value in colors.items():
        if key in string:
            return f"{value}{string}\033[0m"
    return string

def test(data, weights=None, batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.6,
         save_json=False, single_cls=False, augment=False, verbose=False, model=None, dataloader=None,
         save_dir=Path(''), save_txt=False, save_hybrid=False, save_conf=False, plots=False,
         wandb_logger=None, compute_loss=None, half_precision=True, trace=False, is_coco=False, v5_metric=False):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(opt.project) / opt.name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    yolo_model = YOLO(weights)
    model = yolo_model.model.to(device)
    
    half = device.type != 'cpu' and half_precision and not opt.attack
    if half:
        model.half()
    model.eval()
    with open(data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    
    nc = 1 if single_cls else int(data_dict['nc'])
    task = opt.task if opt.task in ('train', 'val', 'test') else 'val'
    
    if opt.attack:
        batch_size = 1
        print(f"{colorstr('Attack:')} Forcing batch_size=1 for attack mode")
    dataset_path = data_dict[task]
    gs = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
    
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = imgsz
    cfg.rect = False
    cfg.cache = False
    cfg.single_cls = single_cls
    cfg.task = 'detect'
    cfg.classes = None
    cfg.fraction = 1.0
    
    dataset = build_yolo_dataset(cfg, dataset_path, batch_size, data_dict, mode='val', rect=False, stride=gs)
    dataloader = build_dataloader(dataset, batch_size, 0, shuffle=False, rank=-1, drop_last=False, pin_memory=True)
    
    print(f"\n{'='*80}")
    print(f"DATALOADER DEBUG:")
    print(f"  Dataset path: {dataset_path}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {imgsz}")
    print(f"{'='*80}\n")

    seen = 0
    names = model.names if hasattr(model, 'names') else {i: f'class{i}' for i in range(nc)}
    confusion_matrix = ConfusionMatrix(names=names)

    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map_val, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    stats, ap, ap_class = [], [], []
    attack_stats_list = []
    
    attacker = None
    summary_file = None
    summary_writer = None
    
    if opt.attack:
        print(f"\n{colorstr('Attack:')} Initializing DCT4OD_mAP attacker...")
        attacker = DCT4OD_mAP(
            model=model,
            epsilon=opt.epsilon,
            max_iter=opt.max_iter,
            step_size=opt.step_size,
            target_map_reduction=opt.target_map_reduction,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            img_size=imgsz,
            names=names,
            freq_type=opt.attack_freq
        )
        summary_path = save_dir / 'attack_summary.csv'
        summary_file = open(summary_path, 'w', newline='', encoding='utf-8', buffering=1)

    for batch_i, batch in enumerate(tqdm(dataloader, desc=s)):
        img = batch['img'].to(device, non_blocking=True)
        img = img.half() if half else img.float()
        img /= 255.0
        batch_idx = batch['batch_idx'].to(device)
        cls = batch['cls'].to(device)
        bboxes = batch['bboxes'].to(device)
        targets = torch.cat([batch_idx.view(-1, 1), cls.view(-1, 1), bboxes], dim=1)
        
        paths = batch['im_file']
        ori_shape = batch['ori_shape']
        ratio_pad = batch['ratio_pad']
        shapes = []
        for i in range(len(paths)):
            ori = ori_shape[i]
            pad = ratio_pad[i]
            if isinstance(ori, torch.Tensor):
                ori = ori.cpu().numpy()
            if isinstance(pad, torch.Tensor):
                pad = pad.cpu().numpy()
            shapes.append((ori, pad))
        
        nb, _, height, width = img.shape

        if attacker and batch_i < opt.attack_imgs:
            try:
                input_dtype = img.dtype
                image_name = Path(paths[0]).name
                
                attack_results = attacker.run(
                    original_image=img.clone(),
                    targets=targets,
                    shapes=shapes,
                    image_path=paths[0],
                    save_dir=save_dir
                )
                
                attack_results['image_name'] = image_name
                attack_stats_list.append(attack_results)
                
                summary_item = {
                    'image_name': image_name,
                    'success': bool(attack_results.get('success')),
                    'attack_time_s': round(attack_results.get('attack_time', 0), 2),
                    'iterations_ran': attack_results.get('iterations_ran'),
                    'map_reduction_percent': round(attack_results.get('map_reduction_percent', 0), 2),
                    'original_map': round(attack_results.get('original_metrics', {}).get('map', 0), 4),
                    'adversarial_map': round(attack_results.get('adversarial_metrics', {}).get('map', 0), 4),
                }
                if summary_writer is None:
                    summary_writer = csv.DictWriter(summary_file, fieldnames=summary_item.keys())
                    summary_writer.writeheader()
                    summary_file.flush()
                    
                summary_writer.writerow(summary_item)
                summary_file.flush()
                
                if attack_results.get('adversarial_image') is not None:
                    adv_img = attack_results['adversarial_image']
                    adv_np = adv_img.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * 255
                    adv_np = adv_np.astype(np.uint8)[:, :, ::-1]
                    save_final_path = save_dir / f"{Path(paths[0]).stem}_adv_FINAL.png"
                    cv2.imwrite(str(save_final_path), adv_np)
                    img = adv_img.to(dtype=input_dtype)

            except Exception as e:
                print(f"\nError attacking image {paths[0]}: {e}")
                import traceback
                traceback.print_exc()

        with torch.no_grad():
            t = time.time()
            out = model(img, augment=augment)
            t0 += time.time() - t
            if isinstance(out, tuple):
                out = out[0]
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, multi_label=True)
            t1 += time.time() - t

        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, 10, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            predn = pred.clone()
            scale_boxes(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])

            correct = torch.zeros(pred.shape[0], 10, dtype=torch.bool, device=device)
            if nl:
                detected = []
                tcls_tensor = labels[:, 0]
                tbox = xywh2xyxy(labels[:, 1:5])
                tbox[:, [0, 2]] *= width
                tbox[:, [1, 3]] *= height
                scale_boxes(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])
                iouv = torch.linspace(0.5, 0.95, 10).to(device)
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)
                    if pi.shape[0]:
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv
                                if len(detected) == nl:
                                    break
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        if opt.attack and batch_i + 1 >= opt.attack_imgs:
            print(f"\n{colorstr('Attack:')} Reached limit of {opt.attack_imgs} images. Stopping.")
            break

    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)
        mp, mr, map50, map_val = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)
    else:
        nt = torch.zeros(1)

    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map_val))
    
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    if opt.attack and summary_file:
        summary_file.close()
        print("\n" + "="*80)
        print(" ADVERSARIAL ATTACK SUMMARY")
        print("="*80)
        print(f"Results saved to: {save_dir}")
        print(f"Summary CSV: {summary_path}")
        if attack_stats_list:
             num = len(attack_stats_list)
             success_rate = sum(1 for x in attack_stats_list if x.get('success')) / num
             print(f"Total images attacked: {num}")
             print(f"Success rate: {success_rate:.2%}")
        print("="*80)

    model.float()
    return (mp, mr, map50, map_val, *(loss.cpu() / len(dataloader)).tolist()), np.zeros(nc), (t0, t1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', type=str, default='yolov8n.pt')
    parser.add_argument('--data', type=str, default='data/tt100k.yaml')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--img-size', type=int, default=2048)
    parser.add_argument('--conf-thres', type=float, default=0.001)
    parser.add_argument('--iou-thres', type=float, default=0.65)
    parser.add_argument('--task', default='val')
    parser.add_argument('--device', default='')
    parser.add_argument('--single-cls', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save-txt', action='store_true')
    parser.add_argument('--save-hybrid', action='store_true')
    parser.add_argument('--save-conf', action='store_true')
    parser.add_argument('--save-json', action='store_true')
    parser.add_argument('--project', default='runs/test')
    parser.add_argument('--name', default='exp')
    parser.add_argument('--exist-ok', action='store_true')
    parser.add_argument('--no-trace', action='store_true')
    parser.add_argument('--v5-metric', action='store_true')
    parser.add_argument('--attack', action='store_true')
    parser.add_argument('--attack_freq', type=str, default='high', choices=['low', 'mid', 'high', 'all'])
    parser.add_argument('--attack_imgs', type=int, default=100)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--iter-atk', type=int, default=1000)
    parser.add_argument('--target_map_reduction', type=float, default=1)

    opt = parser.parse_args()
    opt.max_iter = opt.iter_atk 
    opt.step_size = opt.alpha

    print(colorstr('test: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))

    if opt.task in ('train', 'val', 'test'):
        test(opt.data, opt.weights, opt.batch_size, opt.img_size, opt.conf_thres, opt.iou_thres,
             opt.save_json, opt.single_cls, opt.augment, opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid, save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf, trace=not opt.no_trace, v5_metric=opt.v5_metric)