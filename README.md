1. Download model yolov8: https://drive.google.com/file/d/1EVtO7b5WRiklu7JVMpyxVPQhvqStEZot/view?usp=sharing
2. Download dataset: https://drive.google.com/drive/folders/1Zlg0cycBZs-KUiv_OHNLjXH2Le6XeoNb?usp=sharing
3. Tạo folder data/ và giải nén dataset vào
4. Tạo file tt100k.yaml trong data/ với nội dung như sau:

train: /workspace/ultralytics/data/tt100k/images # 118287 images
val: /workspace/ultralytics/data/tt100k/images # 5000 images
#test: ./coco/test-dev2017.txt # 20288 of 40670 images, submit to https://competitions.codalab.org/competitions/20794

# number of classes

nc: 45

# class names

names: ['i2','i4','i5','il100','il60','il80',
'io','ip','p10','p11','p12','p19','p23','p26',
'p27','p3','p5','p6','pg','ph4','ph4.5','ph5',
'pl100','pl120','pl20','pl30',
'pl40','pl5','pl50','pl60','pl70','pl80','pm20',
'pm30','pm55','pn','pne','po','pr40','w13','w32',
'w55','w57','w59','wo']

5. Chạy pip install -r requirements.txt
6. Chạy auto_attack.py để chạy tự động nhiều thực nghiệm
   Chạy auto_attack_multi.py để chạy tự động và chạy cùng lúc nhiều thực nghiệm
   Chạy test.py để chạy 1 thực nghiệm
