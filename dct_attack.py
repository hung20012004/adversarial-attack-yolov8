import torch
import numpy as np
import time
import random
from tqdm import tqdm
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.metrics import box_iou


def set_seed(seed=0):
    """
    Thiết lập seed ngẫu nhiên cho tất cả thư viện để đảm bảo kết quả lặp lại được
    Mục đích:
    - Đảm bảo mỗi lần chạy với cùng tham số sẽ cho kết quả giống hệt nhau
    Tham số:
        seed (số nguyên): Giá trị hạt giống, mặc định là 0
    """
    random.seed(seed) #Đặt seed cho thư viện random của Python
    np.random.seed(seed) #Đặt seed cho thư viện numpy
    torch.manual_seed(seed) #Đặt seed cho PyTorch trên CPU
    if torch.cuda.is_available(): #Đặt seed cho PyTorch trên GPU nếu có GPU
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False #Tắt chế độ tối ưu hóa để đảm bảo kết quả xác định


def xywh2xyxy(x):
    """
    Chuyển đổi định dạng hộp giới hạn từ [tâm_x, tâm_y, rộng, cao] 
    sang [x1, y1, x2, y2]
    
    Mục đích:
    - YOLOv8 dùng định dạng xywh (tọa độ tâm + kích thước)
    - Nhiều phép tính giao của hộp và vẽ hình cần định dạng xyxy (2 góc)
    
    Tham số:
        x: Mảng tensor hoặc numpy có dạng [..., 4]
           [..., 0] = tọa độ x của tâm
           [..., 1] = tọa độ y của tâm
           [..., 2] = chiều rộng
           [..., 3] = chiều cao
    
    Trả về:
        y: Mảng tensor hoặc numpy có dạng [..., 4]
           [..., 0] = x1 (góc trên bên trái)
           [..., 1] = y1 (góc trên bên trái)
           [..., 2] = x2 (góc dưới bên phải)
           [..., 3] = y2 (góc dưới bên phải)
    
    Công thức:
        x1 = tâm_x - rộng/2
        y1 = tâm_y - cao/2
        x2 = tâm_x + rộng/2
        y2 = tâm_y + cao/2
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """
    Thu phóng tọa độ box giới hạn từ kích thước ảnh đã thay đổi (ảnh1) 
    về kích thước ảnh gốc (ảnh0)
    
    Mục đích:
    - YOLOv8 thay đổi kích thước ảnh về kích thước cố định (ví dụ: 2048x2048) để dự đoán
    - Cần chuyển các dự đoán về tọa độ của ảnh gốc để đánh giá chính xác
    
    Tham số:
        img1_shape: Kích thước ảnh sau khi thay đổi [chiều_cao, chiều_rộng]
        boxes: Hộp giới hạn trên ảnh đã thay đổi, dạng [N, 4+]
        img0_shape: Kích thước ảnh gốc [chiều_cao, chiều_rộng]
        ratio_pad: Bộ (tỷ_lệ, đệm) nếu đã tính trước, None sẽ tự tính
    
    Trả về:
        boxes: Hộp giới hạn đã thu phóng về ảnh gốc
    
    Cách hoạt động:
    1. Tính tỷ lệ thu phóng = min(cao_ảnh1/cao_ảnh0, rộng_ảnh1/rộng_ảnh0)
    2. Tính phần đệm được thêm vào để giữ tỷ lệ khung hình
    3. Trừ phần đệm khỏi tọa độ
    4. Chia cho tỷ lệ để về kích thước gốc
    5. Cắt về trong biên ảnh gốc
    """
    if ratio_pad is None:
        # Tính tỷ lệ - chọn tỷ lệ nhỏ hơn để vừa khít vào ảnh1 mà không cắt
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        # Tính phần đệm ở mỗi bên
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    
    # Loại bỏ phần đệm
    boxes[..., [0, 2]] -= pad[0]  # tọa độ x
    boxes[..., [1, 3]] -= pad[1]  # tọa độ y
    
    # Thu phóng về kích thước gốc
    boxes[..., :4] /= gain
    
    # Cắt để đảm bảo hộp nằm trong biên ảnh
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, img0_shape[1])  # chiều rộng
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, img0_shape[0])  # chiều cao
    return boxes


def compute_ap(recall, precision):
    """
    Tính độ chính xác trung bình (Average Precision - AP) từ các giá trị recall và precision
    
    Mục đích:
    - Độ chính xác trung bình là chỉ số quan trọng nhất để đánh giá object detection
    - Đo diện tích dưới đường cong Precision-Recall
    
    Tham số:
        recall: Mảng các giá trị hồi tưởng (tỷ lệ phát hiện đúng)
        precision: Mảng các giá trị độ chính xác (độ chính xác khi dự đoán)
    
    Trả về:
        ap: Giá trị độ chính xác trung bình (0-1)
        mpre: Mảng độ chính xác đã sửa
        mrec: Mảng hồi tưởng đã sửa
    
    Cách hoạt động:
    1. Thêm điểm đầu (0,1) và điểm cuối (1,0) vào đường cong
    2. Tính độ chính xác đơn điệu (giảm dần từ phải sang trái)
    3. Nội suy độ chính xác tại 101 điểm hồi tưởng (0, 0.01, 0.02, ..., 1.0)
    4. Tính diện tích dưới đường cong bằng phép tích phân hình thang
    
    Ví dụ:
        - ap = 1.0: Mô hình hoàn hảo
        - ap = 0.5: Mô hình trung bình
        - ap = 0.0: Mô hình rất tệ
    """
    # Thêm điểm biên
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    
    # Tính độ chính xác đơn điệu (luôn giảm từ phải sang trái)
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    
    # Tạo 101 điểm hồi tưởng cách đều
    x = np.linspace(0, 1, 101)
    
    # Tính độ chính xác trung bình bằng tích phân hình thang
    try:
        ap = np.trapezoid(np.interp(x, mrec, mpre), x)  #syntax numpy >= 2.0
    except AttributeError:
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # numpy < 2.0
    return ap, mpre, mrec


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir=None, names=(), eps=1e-16):
    """
    Tính độ chính xác trung bình cho từng lớp riêng biệt->vẽ đường
    
    Mục đích:
    - Đánh giá hiệu suất mô hình trên từng loại vật thể
    - Tính độ chính xác trung bình tổng hợp = trung bình các lớp
    
    Tham số:
        tp: Mảng dương tính thật(True positive array), dạng [N_predictions, 10] cho các ngưỡng giao(IoU thresholds) 0.5:0.95
        conf: Điểm tin cậy của dự đoán, dạng [N_predictions]
        pred_cls: Lớp được dự đoán, dạng [N_predictions]
        target_cls: Lớp thực tế, dạng [N_targets]
        plot: Có vẽ đường cong hay không
        save_dir: Thư mục lưu hình vẽ
        names: Từ điển mapping class_id -> class_name
        eps: Epsilon nhỏ để tránh chia cho 0
    
    Trả về:
        tp: True positives tại điểm F1 tốt nhất
        fp: False positives tại điểm F1 tốt nhất
        p: Độ chính xác tại điểm F1 tốt nhất
        r: Hồi tưởng tại điểm F1 tốt nhất
        f1: Điểm F1 tại điểm tốt nhất
        ap: Độ chính xác trung bình cho mỗi lớp, dạng [số_lớp, 10]
        unique_classes: Các lớp có trong dữ liệu thực
    
    Cách hoạt động:
    1. Sắp xếp dự đoán theo độ tin cậy giảm dần
    2. Với mỗi lớp:
       - True positives(TP) và false positives(FP) tích lũy
       - Tính hồi tưởng = dương_tính_thật(TP) / (dương_tính_thật(TP) + âm_tính_giả(FP))
       - Tính độ chính xác = dương_tính_thật / (dương_tính_thật + dương_tính_giả)
       - Nội suy đường cong tại 1000 điểm
       - Tính độ chính xác trung bình cho 10 ngưỡng giao
    3. Tính F1 = 2*P*R/(P+R) và chọn ngưỡng tốt nhất
    """
    # Sắp xếp theo độ tin cậy giảm dần (dự đoán tự tin nhất trước)
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    
    # Lấy các lớp duy nhất và số lượng thực của mỗi lớp
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # số lớp
    
    # Khởi tạo mảng để lưu chỉ số
    px, py = np.linspace(0, 1, 1000), []  # 1000 điểm hồi tưởng để nội suy
    ap = np.zeros((nc, tp.shape[1]))  # độ chính xác trung bình cho mỗi lớp, mỗi ngưỡng giao
    p = np.zeros((nc, 1000))  # đường cong độ chính xác
    r = np.zeros((nc, 1000))  # đường cong hồi tưởng
    
    # Tính độ chính xác trung bình cho từng lớp
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c  # mặt nạ cho dự đoán của lớp này
        n_l = nt[ci]  # số dữ liệu thực của lớp này
        n_p = i.sum()  # số dự đoán của lớp này
        
        if n_p == 0 or n_l == 0:
            continue
        
        # Tính dương tính giả và dương tính thật tích lũy
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)
        
        # Hồi tưởng = dương_tính_thật / tổng_số_thực
        recall = tpc / (n_l + eps)
        
        # Nội suy đường cong hồi tưởng tại 1000 điểm
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)
        
        # Độ chính xác = dương_tính_thật / (dương_tính_thật + dương_tính_giả)
        precision = tpc / (tpc + fpc)
        
        # Nội suy đường cong độ chính xác tại 1000 điểm
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)
        
        # Tính độ chính xác trung bình cho 10 ngưỡng giao (0.5, 0.55, ..., 0.95)
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
    
    # Tính điểm F1 = 2*P*R/(P+R)
    f1 = 2 * p * r / (p + r + eps)
    
    # Tìm điểm có điểm F1 trung bình cao nhất
    i = f1.mean(0).argmax()
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt.reshape((-1, 1))).round()  # dương tính thật tại điểm F1 tốt nhất
    
    return tp, fpc, p, r, f1, ap, unique_classes.astype(int)


def create_frequency_mask(H, W, freq_type, r_h=0.2, r_w=0.2, d_min=0.2, d_max=0.6, device='gpu'):
    """
    Tạo mặt nạ:
    - Tần số thấp: Thông tin tổng thể, màu sắc, hình dạng lớn
    - Tần số tb: Chi tiết vừa phải, cạnh, kết cấu
    - Tần số cao: Chi tiết mịn, nhiễu, cạnh sắc nét
    
    Tham số:
        H, W: Kích thước ảnh (chiều cao, chiều rộng)
        freq_type: Loại tần số cần tấn công
            - 'low': Tấn công tần số thấp (góc trên trái)
            - 'high': Tấn công tần số cao (phần bù low)
            - 'mid': Tấn công tần số tba (vòng tròn từ d_min đến d_max)
            - 'all': Tấn công toàn bộ tần số
        r_h, r_w: Tỷ lệ vùng tần số thấp
        d_min, d_max: giới hạn cho mid
        device: CPU hoặc GPU
    
    Trả về:
        mask: Mặt nạ nhị phân dạng [1, 1, H, W]
              1 = tần số được tấn công, 0 = tần số bỏ qua
    """
    mask = torch.zeros((1, 1, H, W), device=device, dtype=torch.float32)
    
    if freq_type == 'low':
        # Tần số thấp: góc trên trái của phổ
        size_h, size_w = int(r_h * H), int(r_w * W)
        mask[:, :, :size_h, :size_w] = 1
        
    elif freq_type == 'high':
        # Tần số cao: toàn bộ trừ góc trên trái
        size_h, size_w = int(r_h * H), int(r_w * W)
        mask[:, :, :, :] = 1
        mask[:, :, :size_h, :size_w] = 0
        
    elif freq_type == 'mid':
        # Tần số tb: vòng tròn từ d_min đến d_max
        # Tạo lưới tọa độ chuẩn hóa [0, 1]
        i_coords = torch.arange(H, device='cpu').float().unsqueeze(1) / H
        j_coords = torch.arange(W, device='cpu').float().unsqueeze(0) / W
        
        # Tính khoảng cách từ góc trên trái (khoảng cách Euclid)
        dist = torch.sqrt(i_coords**2 + j_coords**2)
        
        # Chọn vùng tần số giữa
        mid_mask = (dist > d_min) & (dist < d_max)
        mask[0, 0] = mid_mask.to(device)
        
    elif freq_type == 'all':
        # Tấn công toàn bộ tần số
        mask[:, :, :, :] = 1
        
    else:
        raise ValueError(f"freq_type phải là 'low', 'mid', 'high', hoặc 'all'")
    
    return mask


# Bộ nhớ đệm để lưu ma trận biến đổi cosin, tránh tính lại nhiều lần
_DCT_MATRIX_CACHE = {}


class DCT4OD_mAP:
    """
    1. Chuyển ảnh sang miền tần số (miền biến đổi cosin)
    2. Thêm nhiễu vào các thành phần tần số được chọn (thấp/giữa/cao)
    3. Chuyển ngược về miền không gian
    4. Giữ nhiễu trong ngưỡng epsilon để không nhìn thấy bằng mắt thường
    """
    
    def __init__(self, model, epsilon, max_iter, step_size, target_map_reduction,
                 conf_thres, iou_thres, img_size, names=None, freq_type='low',
                 r_h=0.2, r_w=0.2, d_min=0.2, d_max=0.6):
        """
        Khởi tạo bộ tấn công
        Tham số:
            model: Mô hình YOLOv8
            epsilon: Ngưỡng nhiễu tối đa (0-1 hoặc 0-255) chuẩn hóa về 0-1
                    Vd: epsilon=8/255 = 0.031
            max_iter: Số vòng lặp tối đa
            step_size: Kích thước bước mỗi vòng lặp (0-1 hoặc 0-255)
            target_map_reduction: Mức giảm độ chính xác mong muốn (0-1)
                    Vd: 0.3 = giảm 30%
            conf_thres: Ngưỡng tin cậy cho lọc hộp
            iou_thres: Ngưỡng giao cho lọc hộp
            img_size: Kích thước ảnh đầu vào
            names: mảng mã_lớp - tên_lớp
            freq_type: Loại tần số tấn công ('low', 'mid', 'high', 'all')
            r_h, r_w: Tỷ lệ vùng tần số thấp
            d_min, d_max: Khoảng cách cho tần số tb
        """
        self.model = model
        
        # Chuẩn hóa epsilon và stepsize về [0, 1]
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
        
        # In thông tin cấu hình
        print(f"\n{'='*80}")
        print(f"Initializing Full-DCT Attack (YOLOv8)")
        print(f"  Image Size: {img_size}×{img_size}")
        print(f"  Frequency: {freq_type}")
        print(f"  Epsilon: {epsilon}")
        print(f"  Step Size: {step_size}")
        print(f"  Max Iterations: {max_iter}")
        print(f"{'='*80}\n")
        
        # Tạo ma trận biến đổi cosin (lưu cache để tái sử dụng)
        self.dct_matrix = self._create_dct_matrix(img_size).to(self.device, dtype=self.dtype)
        
        # Tạo mặt nạ tần số
        self.freq_mask = create_frequency_mask(
            H=img_size, W=img_size, freq_type=freq_type,
            r_h=r_h, r_w=r_w, d_min=d_min, d_max=d_max,
            device=self.device
        )
        
        # Hiển thị tỷ lệ tần số được tấn công
        active_ratio = (self.freq_mask.sum() / (img_size * img_size)) * 100
        print(f"Thành phần tần số hoạt động: {active_ratio:.1f}%\n")

    def _extract_predictions(self, model_output):
        """
        Trích xuất tensor dự đoán từ đầu ra của YOLOv8
        
        Mục đích:
        - YOLOv8 có thể trả về nhiều định dạng khác nhau: bộ, từ điển, tensor
        - Cần trích xuất tensor dự đoán để xử lý
        
        Tham số:
            model_output: Đầu ra từ model.forward()
                         - Bộ: (dự_đoán, mất_mát) hoặc (dự_đoán,)
                         - Từ điển: {'pred': tensor, ...}
                         - Tensor: dự đoán trực tiếp
        
        Trả về:
            predictions: Tensor chứa dự đoán
        
        Cách hoạt động:
        1. Nếu là bộ: lấy phần tử đầu tiên
        2. Nếu là từ điển: tìm khóa 'pred', 'predictions', 'output', hoặc 'logits'
        3. Nếu là tensor: trả về trực tiếp
        """
        if isinstance(model_output, tuple): #nếu là tuple lấy phần tử đầu tiên
            return model_output[0]
            
        elif isinstance(model_output, dict): #nếu là dict lấy theo các key 'pred', 'predictions', 'output', 'logits'
            for key in ['pred', 'predictions', 'output', 'logits']:
                if key in model_output and isinstance(model_output[key], torch.Tensor):
                    return model_output[key]
            # Nếu là tensor: trả về trực tiếp
            for v in model_output.values():
                if isinstance(v, torch.Tensor):
                    return v
            raise ValueError(f"Cannot extract tensor from dict with keys: {model_output.keys()}")
        else:
            return model_output

    def _create_dct_matrix(self, n):
        """
        Tạo ma trận biến đổi cosin rời rạc dùng để chuyển ảnh sang miền tần số
        Ma trận này có kích thước n*n lưu vào bộ nhớ đệm để tái sử dụng
        
        Trả về:
            dct_m: Ma trận biến đổi cosin dạng [n, n]
        
        Công thức biến đổi cosin loại II (chuẩn):
            D[k,i] = sqrt(1/n)                          nếu k=0
                   = sqrt(2/n) * cos(π*k*(2i+1)/(2n))  nếu k>0
        
        Trong đó:
            k: chỉ số tần số (0 đến n-1)
            i: chỉ số không gian (0 đến n-1)
        
        Ví dụ với n=4:
            Hàng 0: Thành phần một chiều (tần số 0)
            Hàng 1,2,3: Tần số tăng dần
        """
        cache_key = (n, str(self.device), str(self.dtype))
        
        if cache_key in _DCT_MATRIX_CACHE:
            print(f"Using cached DCT matrix ({n}×{n})")
            return _DCT_MATRIX_CACHE[cache_key]
        
        print(f"Tạo ma trận biến đổi cosin ({n}×{n})... ", end='', flush=True)
        start_time = time.time()
        
        # Tạo lưới chỉ số
        k = torch.arange(n, dtype=self.dtype, device='cpu').view(-1, 1)  # [n, 1]
        i = torch.arange(n, dtype=self.dtype, device='cpu').view(1, -1)  # [1, n]
        
        # Tính ma trận biến đổi cosin theo công thức
        dct_m = torch.where(
            k == 0,
            # Hàng đầu tiên (thành phần một chiều)
            torch.sqrt(torch.tensor(1.0 / n, dtype=self.dtype)),
            # Các hàng còn lại (thành phần xoay chiều)
            torch.sqrt(torch.tensor(2.0 / n, dtype=self.dtype)) * 
            torch.cos(torch.pi * k * (2 * i + 1) / (2 * n))
        )
        
        # Lưu cache
        _DCT_MATRIX_CACHE[cache_key] = dct_m
        print(f"Xong! ({time.time() - start_time:.2f}s)")
        return dct_m

    def full_dct_2d(self, x):
        """
        Thực hiện biến đổi cosin 2 chiều trên ảnh
        
        Mục đích:
        - Chuyển ảnh từ miền không gian sang miền tần số
        - Không gian: giá trị điểm ảnh, Tần số: các thành phần tần số
        
        Tham số:
            x: Tensor đầu vào dạng [B, C, H, W]
               B = kích thước lô
               C = kênh (3 cho màu)
               H, W = chiều cao, chiều rộng
        
        Trả về:
            x_dct: Hệ số biến đổi cosin dạng [B, C, H, W]
        
        Công thức biến đổi cosin 2 chiều:
            X_dct = D * X * D^T
        
        Trong đó:
            D: Ma trận biến đổi cosin [H, H]
            X: Ảnh đầu vào [H, W]
            D^T: Chuyển vị của D
        
        Cách hoạt động:
        1. Đổi hình [B,C,H,W] → [B*C, H, W] để xử lý từng kênh
        2. Nhân bên trái: D * X (biến đổi theo hàng)
        3. Nhân bên phải: (D*X) * D^T (biến đổi theo cột)
        4. Đổi hình về [B, C, H, W]
        
        Ý nghĩa:
        - Góc trên trái: Tần số thấp (thành phần một chiều, màu sắc tổng thể)
        - Góc dưới phải: Tần số cao (chi tiết mịn, cạnh)
        """
        B, C, H, W = x.shape
        x_2d = x.view(-1, H, W)  # [B*C, H, W]
        
        # D * X
        x_dct = torch.matmul(self.dct_matrix, x_2d)
        
        # (D * X) * D^T
        x_dct = torch.matmul(x_dct, self.dct_matrix.T)
        
        return x_dct.view(B, C, H, W)

    def full_idct_2d(self, x_dct):
        """
        Thực hiện biến đổi cosin ngược 2 chiều
        
        Mục đích:
        - Chuyển từ miền tần số về miền không gian
        - Dùng sau khi thêm nhiễu vào hệ số biến đổi cosin
        
        Tham số:
            x_dct: Hệ số biến đổi cosin dạng [B, C, H, W]
        
        Trả về:
            x: Ảnh tái tạo dạng [B, C, H, W]
        
        Công thức biến đổi cosin ngược 2 chiều:
            X = D^T * X_dct * D
        
        Cách hoạt động:
        1. Đổi hình [B,C,H,W] → [B*C, H, W]
        2. Nhân bên trái: D^T * X_dct
        3. Nhân bên phải: (D^T * X_dct) * D
        4. Đổi hình về [B, C, H, W]
        
        Tính chất:
        - Biến đổi ngược(Biến đổi(X)) = X (phép biến đổi đảo ngược được)
        - Cho phép sửa đổi miền tần số rồi quay về miền không gian
        """
        B, C, H, W = x_dct.shape
        x_dct_2d = x_dct.view(-1, H, W)  # [B*C, H, W]
        
        # D^T * X_dct
        x = torch.matmul(self.dct_matrix.T, x_dct_2d)
        
        # (D^T * X_dct) * D
        x = torch.matmul(x, self.dct_matrix)
        
        return x.view(B, C, H, W)

    def compute_map(self, predictions, targets, img_tensor, shapes):
        """
        Tính độ chính xác trung bình tổng hợp cho dự đoán YOLOv8
        
        Mục đích:
        - Đánh giá độ chính xác của mô hình trên ảnh đối nghịch
        - Độ chính xác trung bình tổng hợp là chỉ số chính để đo hiệu quả của tấn công
        
        Tham số:
            predictions: Đầu ra mô hình, có thể là bộ/từ điển/tensor
            targets: Nhãn thực dạng [N, 6]
                    [mã_ảnh, lớp, x, y, w, h]
            img_tensor: Ảnh đầu vào dạng [B, C, H, W]
            shapes: Danh sách (hình_gốc, (tỷ_lệ, đệm)) cho mỗi ảnh
        
        Trả về:
            từ điển: {'map': độ_chính_xác@0.5:0.95, 'map50': độ_chính_xác@0.5}
        
        Cách hoạt động:
        1. Trích xuất tensor dự đoán
        2. Áp dụng lọc hộp để loại bỏ hộp trùng lặp
        3. Với mỗi ảnh:
           - Khớp dự đoán với thực tế bằng giao của hộp
           - Đánh dấu dương tính thật nếu giao > ngưỡng
        4. Tính độ chính xác trung bình cho từng lớp
        5. Trung bình độ chính xác của tất cả lớp = độ chính xác trung bình tổng hợp
        
        Độ chính xác@0.5:0.95:
        - Trung bình của độ chính xác tại 10 ngưỡng giao (0.5, 0.55, ..., 0.95)
        - Chỉ số khắt khe hơn độ chính xác@0.5
        """
        try:
            # Trích xuất dự đoán từ đầu ra mô hình
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            elif isinstance(predictions, dict):
                for key in ['pred', 'predictions', 'output']:
                    if key in predictions and isinstance(predictions[key], torch.Tensor):
                        predictions = predictions[key]
                        break
            
            predictions = predictions.detach()
            
            # Áp dụng lọc hộp không cực đại
            # Loại bỏ các hộp trùng lặp, giữ lại hộp có độ tin cậy cao nhất
            output = non_max_suppression(
                predictions.clone(), 
                self.conf_thres,  # Ngưỡng tin cậy
                self.iou_thres,   # Ngưỡng giao
                multi_label=True  # Cho phép 1 hộp dự đoán nhiều lớp
            )
            
            # 10 ngưỡng giao từ 0.5 đến 0.95
            iouv = torch.linspace(0.5, 0.95, 10).to(self.device)
            stats = []
            
            # Xử lý từng ảnh trong lô
            for si, pred in enumerate(output):
                # Lấy nhãn thực cho ảnh này
                labels = targets[targets[:, 0] == si, 1:]  # [lớp, x, y, w, h]
                nl = len(labels)  # Số hộp thực
                tcls = labels[:, 0].tolist() if nl else []  # Lớp thực
                
                # Nếu không có dự đoán
                if len(pred) == 0:
                    if nl:
                        # Có thực nhưng không dự đoán → toàn bộ âm tính giả
                        stats.append((
                            torch.zeros(0, 10, dtype=torch.bool),
                            torch.Tensor(),
                            torch.Tensor(),
                            tcls
                        ))
                    continue
                
                # Khởi tạo ma trận đúng [N_dự_đoán, 10_ngưỡng_giao]
                correct = torch.zeros(pred.shape[0], 10, dtype=torch.bool, device=self.device)
                
                if nl:
                    tcls_tensor = labels[:, 0]
                    
                    # Chuyển hộp thực từ xywh sang xyxy
                    tbox = xywh2xyxy(labels[:, 1:5])
                    
                    # Thu phóng hộp thực về kích thước ảnh gốc
                    scale_boxes(img_tensor[si].shape[1:], tbox, shapes[si][0], shapes[si][1])
                    
                    # Sao chép dự đoán và thu phóng về ảnh gốc
                    predn = pred.clone()
                    scale_boxes(img_tensor[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])
                    
                    detected = []  # Danh sách hộp thực đã được phát hiện
                    
                    # Với mỗi lớp
                    for cls in torch.unique(tcls_tensor):
                        # Chỉ số của hộp thực thuộc lớp này
                        ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)
                        
                        # Chỉ số của hộp dự đoán thuộc lớp này
                        pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)
                        
                        if pi.shape[0]:
                            # Tính giao giữa dự đoán và thực
                            # ious dạng: [N_dự_đoán, N_thực]
                            ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)
                            
                            # Với mỗi dự đoán có giao > 0.5
                            for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                d = ti[i[j]]  # Hộp thực được khớp
                                
                                if d.item() not in detected:
                                    detected.append(d.item())
                                    
                                    # Đánh dấu đúng cho các ngưỡng giao
                                    correct[pi[j]] = ious[j] > iouv
                                    
                                    if len(detected) == nl:
                                        break  # Đã khớp hết hộp thực
                
                # Lưu thống kê: (đúng, độ_tin_cậy, lớp_dự_đoán, lớp_thực)
                stats.append((
                    correct.cpu().detach(),
                    pred[:, 4].cpu().detach(),  # Điểm tin cậy
                    pred[:, 5].cpu().detach(),  # Lớp dự đoán
                    tcls
                ))
            
            # Tính độ chính xác trung bình nếu có ít nhất 1 dương tính thật
            if len(stats) and stats[0][0].any():
                # Ghép thống kê từ tất cả ảnh
                stats_np = [np.concatenate(x, 0) for x in zip(*stats)]
                
                # Tính độ chính xác trung bình cho mỗi lớp
                tp, fp, p, r, f1, ap, ap_class = ap_per_class(
                    *stats_np, 
                    plot=False, 
                    names=self.names
                )
                
                return {
                    'map': float(ap.mean()),      # độ_chính_xác@0.5:0.95
                    'map50': float(ap[:, 0].mean())  # độ_chính_xác@0.5
                }
            
            # Không có dương tính thật → độ chính xác = 0
            return {'map': 0.0, 'map50': 0.0}
        
        except Exception as e:
            print(f"Cảnh báo trong compute_map: {e}")
            return {'map': 0.0, 'map50': 0.0}

    def run(self, original_image, targets, shapes, image_path, save_dir):
        """
        Thực hiện tấn công đối nghịch dựa trên biến đổi cosin
        
        Mục đích:
        - Tạo ảnh đối nghịch để giảm độ chính xác của YOLOv8
        - Giữ nhiễu trong ngưỡng epsilon để không nhìn thấy
        
        Tham số:
            original_image: Tensor ảnh gốc [B, C, H, W], phạm vi [0, 1]
            targets: Nhãn thực [N, 6] = [mã_ảnh, lớp, x, y, w, h]
            shapes: Danh sách (hình_gốc, (tỷ_lệ, đệm))
            image_path: Đường dẫn ảnh (để ghi nhật ký)
            save_dir: Thư mục lưu kết quả
        
        Trả về:
            từ điển: {
                'success': Tấn công thành công hay không
                'adversarial_image': Ảnh đối nghịch tốt nhất
                'original_metrics': Độ chính xác của ảnh gốc
                'adversarial_metrics': Độ chính xác của ảnh đối nghịch
                'map_reduction_percent': % giảm độ chính xác
                'iterations_ran': Số vòng lặp đã chạy
                'attack_time': Thời gian tấn công (giây)
            }
        
        Thuật toán (Giảm gradient chiếu trong miền biến đổi cosin):
        
        1. Tính độ chính xác cơ sở trên ảnh gốc
        2. Với mỗi vòng lặp:
           a. Lan truyền xuôi: tính mất mát (tối đa hóa lỗi phát hiện)
           b. Lan truyền ngược: tính gradient của mất mát theo đầu vào
           c. Biến đổi cosin gradient
           d. Lọc gradient (chỉ giữ thành phần tần số được chọn)
           e. Biến đổi cosin ngược về miền không gian
           f. Cập nhật ảnh đối nghịch theo hướng gradient
           g. Chiếu về hình cầu epsilon: cắt nhiễu trong [-ε, +ε]
           h. Cắt ảnh trong [0, 1]
        3. Định kỳ kiểm tra độ chính xác, lưu ảnh đối nghịch tốt nhất
        4. Dừng khi đạt mục tiêu hoặc hết vòng lặp
        
        Công thức cập nhật:
            grad_dct = Biến_đổi_cosin(∇_x Mất_mát)
            grad_dct_lọc = grad_dct ⊙ mặt_nạ
            nhiễu = Biến_đổi_cosin_ngược(grad_dct_lọc)
            x_đối_nghịch = x_đối_nghịch - bước_nhảy * nhiễu / ||nhiễu||
            x_đối_nghịch = cắt(x_đối_nghịch, x_gốc - ε, x_gốc + ε)
            x_đối_nghịch = cắt(x_đối_nghịch, 0, 1)
        """
        start_time = time.time()
        nb, _, height, width = original_image.shape
        
        # Lưu kiểu dữ liệu gốc của mô hình
        original_model_dtype = next(self.model.parameters()).dtype
        
        # Chuyển mô hình sang số thực 32 bit (cần thiết để tính gradient)
        self.model.float()
        for param in self.model.parameters():
            param.data = param.data.float()
        for buffer in self.model.buffers():
            buffer.data = buffer.data.float()
        
        original_image = original_image.detach().float()
        
        # Thu phóng mục tiêu về tọa độ điểm ảnh
        # Mục tiêu ban đầu ở dạng chuẩn hóa [0, 1]
        targets_scaled = targets.clone().detach()
        if targets_scaled.shape[1] >= 6:
            targets_scaled[:, 2:6] *= torch.Tensor([width, height, width, height]).to(self.device)
        
        # ==================== Tính độ chính xác cơ sở ====================
        self.model.eval()
        with torch.no_grad():
            model_output = self.model(original_image)
            orig_preds = self._extract_predictions(model_output)
            metrics_orig = self.compute_map(orig_preds, targets_scaled, original_image, shapes)
            orig_map = metrics_orig['map']
        
        # Tính độ chính xác mục tiêu cần đạt
        target_map = orig_map * (1 - self.target_map_reduction)
        print(f"Độ chính xác gốc: {orig_map:.4f} | Mục tiêu: {target_map:.4f}")
        
        # Khởi tạo ảnh đối nghịch
        adversarial_image = original_image.clone().detach().requires_grad_(True)
        best_map = orig_map
        best_adv = original_image.clone().detach()
        
        # Tần suất kiểm tra độ chính xác (mỗi 2% tiến trình)
        check_interval = max(1, self.max_iter // 50)
        
        pbar = tqdm(range(self.max_iter), desc="Đang tấn công", ncols=100)
        
        # ==================== Vòng lặp tấn công ====================
        for i in pbar:
            self.model.eval()
            
            # Lan truyền xuôi
            model_output = self.model(adversarial_image)
            
            # Trích xuất dự đoán
            try:
                preds = self._extract_predictions(model_output)
            except Exception as e:
                print(f"\nLỗi trích xuất dự đoán tại vòng {i}: {e}")
                break
            
            # ==================== Tính mất mát ====================
            # Mục tiêu: Giảm thiểu độ tin cậy/độ vật thể để giảm phát hiện
            # Dự đoán YOLOv8 có nhiều định dạng khác nhau
            if preds.dim() >= 3:
                if preds.shape[1] > preds.shape[2]:
                    # Dạng [B, đặc_trưng, H, W] → chuyển vị
                    preds = preds.transpose(1, 2).transpose(2, 3)
                    loss = preds[..., 4].sum()  # Điểm độ vật thể
                else:
                    # Dạng [B, N, đặc_trưng]
                    loss = preds[..., 4].sum() if preds.shape[-1] > 4 else preds.max()
            else:
                loss = preds.sum()
            
            # Kiểm tra gradient
            if not loss.requires_grad:
                print(f"\nMất mát không có gradient tại vòng {i}")
                break
            
            if loss.item() == 0:
                print(f"\nMất mát bằng 0 tại vòng {i}, dừng")
                break
            
            # ==================== Lan truyền ngược ====================
            loss.backward()

            if adversarial_image.grad is None:
                print(f"\nKhông có gradient tại vòng {i}")
                break

            # ==================== Cập nhật ảnh đối nghịch ====================
            with torch.no_grad():
                # 1. Lấy gradient
                grad = adversarial_image.grad.clone()
                
                # 2. Biến đổi cosin gradient
                grad_dct = self.full_dct_2d(grad)
                
                # 3. Áp dụng mặt nạ tần số
                grad_dct_masked = grad_dct * self.freq_mask
                
                # 4. Biến đổi cosin ngược về miền không gian
                noise = self.full_idct_2d(grad_dct_masked)
                
                # 5. Chuẩn hóa nhiễu
                norm = torch.linalg.norm(noise) + 1e-10
                
                # 6. Bước giảm gradient
                adversarial_image.data -= self.step_size * (noise / norm)
                
                # 7. Chiếu về hình cầu epsilon
                pert = (adversarial_image.data - original_image.data).clamp(-self.epsilon, self.epsilon)
                adversarial_image.data = (original_image.data + pert).clamp(0, 1)

            # Xóa gradient
            adversarial_image.grad.zero_()
            self.model.zero_grad()
            
            # Giải phóng bộ nhớ
            if i % 10 == 0:
                torch.cuda.empty_cache()

            # ==================== Đánh giá độ chính xác ====================
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
                    
                    # Lưu ảnh đối nghịch tốt nhất
                    if curr_map < best_map:
                        best_map = curr_map
                        best_adv = adversarial_image.clone().detach()
                    
                    # Cập nhật thanh tiến trình
                    pbar.set_postfix({
                        'Độ_chính_xác': f'{curr_map:.4f}',
                        'tốt_nhất': f'{best_map:.4f}',
                        'mục_tiêu': f'{target_map:.4f}'
                    })
                    
                    # Dừng sớm nếu đạt mục tiêu
                    if curr_map <= target_map:
                        print(f"\n✓ Đạt mục tiêu tại vòng {i+1}")
                        break

        pbar.close()
        
        # ==================== Tính chỉ số ====================
        attack_time = time.time() - start_time
        reduction = ((orig_map - best_map) / (orig_map + 1e-9)) * 100
        
        del adversarial_image
        
        # Khôi phục kiểu dữ liệu mô hình
        if original_model_dtype == torch.float16:
            self.model.half()
            for param in self.model.parameters():
                param.data = param.data.half()
            for buffer in self.model.buffers():
                buffer.data = buffer.data.half()
        
        torch.cuda.empty_cache()
        
        success = best_map <= target_map
        
        # ==================== In kết quả ====================
        print(f"\n{'='*80}")
        print(f"Tấn công {'THÀNH CÔNG ✓' if success else 'THẤT BẠI ✗'}")
        print(f"  Độ chính xác gốc: {orig_map:.4f}")
        print(f"  Độ chính xác cuối: {best_map:.4f}")
        print(f"  Giảm: {reduction:.1f}%")
        print(f"  Thời gian: {attack_time:.1f}s")
        print(f"  Vòng lặp: {i + 1}/{self.max_iter}")
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