import cv2
import os
from pathlib import Path
from ultralytics import YOLO


class YOLOv11Cropper:
    def __init__(self, model_path, conf=0.25, iou=0.7):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        print(f"✅ 加载模型: {model_path} (类型: {type(self.model).__name__})")

    def crop_objects(self, input_dir, output_dir):
        """执行裁剪操作"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 遍历所有图像文件
        for img_path in Path(input_dir).rglob('*'):
            if not img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                continue

            try:
                # 读取图像
                img = cv2.imread(str(img_path))
                if img is None:
                    raise ValueError("无法读取图像")

                # 执行检测
                results = self.model.predict(
                    img,
                    conf=self.conf,
                    iou=self.iou,
                    save=False,
                    verbose=False
                )

                # 处理检测结果
                if not results[0].boxes:
                    print(f"🚫 {img_path.name} 未检测到目标")
                    continue

                # 裁剪并保存
                self._process_image(img, results[0], img_path, output_dir)

            except Exception as e:
                print(f"❌ 处理 {img_path.name} 时出错: {str(e)}")
                continue

    def _process_image(self, img, results, img_path, output_dir):
        """单张图像处理核心逻辑"""
        # 创建输出子目录
        base_name = img_path.stem
        output_subdir = output_dir

        # 遍历所有检测框
        for idx, box in enumerate(results.boxes):
            try:
                # 获取坐标（转换为整数）
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # 验证坐标有效性
                if x2 <= x1 or y2 <= y1:
                    print(f"⚠️ {img_path.name} 检测框坐标无效")
                    continue

                # 裁剪目标区域
                cropped = img[y1:y2, x1:x2]

                # 生成保存路径
                save_path = output_subdir / f"{base_name}_{idx}.jpg"

                # 保存裁剪结果
                cv2.imwrite(str(save_path), cropped)
                print(f"✅ 保存裁剪结果: {save_path} ({idx + 1}/{len(results.boxes)})")

            except Exception as e:
                print(f"❌ 裁剪 {img_path.name} 时出错: {str(e)}")
                continue


if __name__ == "__main__":
    # 配置参数
    MODEL_PATH = r"C:\Users\srt69\Desktop\weights\best.pt"  # 模型路径
    INPUT_DIR = r"E:\data\右前门切水条\分类\分类C"  # 输入图像目录
    OUTPUT_DIR = r"E:\data\右前门切水条\分类\分类C_crop"  # 输出目录

    # 创建裁剪器实例
    cropper = YOLOv11Cropper(MODEL_PATH)

    # 执行裁剪操作
    cropper.crop_objects(INPUT_DIR, OUTPUT_DIR)