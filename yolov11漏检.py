import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import cv2
import time


class YOLOv11LeakDetectorWithCopy:
    def __init__(self, model_path, leak_dir="leak_images", conf_low=0.05, conf_high=0.25):
        self.model = YOLO(model_path)
        self.leak_dir = Path(leak_dir)
        self.conf_low = conf_low
        self.conf_high = conf_high

        # 创建漏检专用目录
        self.leak_dir.mkdir(exist_ok=True)
        self.processed_dir = self.leak_dir / "processed"
        self.processed_dir.mkdir(exist_ok=True)

    def _copy_leak_image(self, src_path, leak_id):
        """复制漏检图像到专用目录"""
        try:
            # 保留原始文件名
            dest_path = self.leak_dir / src_path.name
            # 防止重复覆盖
            if dest_path.exists():
                base, ext = os.path.splitext(dest_path)
                dest_path = f"{base}_{leak_id}{ext}"

            shutil.copy(src_path, dest_path)
            print(f"已复制漏检图像: {dest_path}")
            return True
        except Exception as e:
            print(f"复制失败: {src_path} -> 错误: {e}")
            return False

    def process_image(self, img_path):
        """处理单张图像并复制漏检样本"""
        start_time = time.time()
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[跳过] 无法读取图像: {img_path}")
            return

        # 双阶段检测
        results_low = self.model.predict(img, conf=self.conf_low, verbose=False)[0]
        results_high = self.model.predict(img, conf=self.conf_high, verbose=False)[0]

        # 提取检测框
        boxes_low = results_low.boxes.xyxy.cpu().numpy()
        boxes_high = results_high.boxes.xyxy.cpu().numpy()

        # 漏检判定
        if len(boxes_low) > 0 and len(boxes_high) == 0:
            leak_id = len(os.listdir(self.leak_dir)) + 1
            self._copy_leak_image(img_path, leak_id)
            print(f"[漏检] 发现新漏检样本: {img_path} -> leak_{leak_id}.jpg")
            return

        process_time = time.time() - start_time
        print(f"[正常] 处理完成: {img_path} ({process_time:.2f}s)")

    def batch_process(self, input_dir):
        """批量处理图像"""
        image_paths = list(Path(input_dir).rglob('*.[jp][pn]g'))
        print(f"发现 {len(image_paths)} 张待检测图像")

        for img_path in image_paths:
            self.process_image(img_path)


if __name__ == "__main__":
    # 初始化检测器
    detector = YOLOv11LeakDetectorWithCopy(
        model_path=r'E:\data\3线切水条-089\best.pt',  # 替换为你的模型路径
        leak_dir="leak_samples",  # 漏检图像保存目录
        conf_low=0.05,  # 低阈值检测
        conf_high=0.6  # 高阈值检测
    )

    # 执行批量检测
    detector.batch_process(
        input_dir=r'E:\data\3线切水条\896W-切水条\新建文件夹'  # 替换为你的图像目录
    )