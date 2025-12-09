import os
import cv2
from ultralytics import YOLO
from pathlib import Path


class YOLOv11FolderInference:
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.7):
        """
        初始化YOLOv11推理器
        :param model_path: 模型文件路径
        :param conf_threshold: 置信度阈值
        :param iou_threshold: IOU阈值
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        print(f"✅ 模型加载成功: {model_path}")

    def process_folder(self, input_folder, output_base_folder=None):
        """
        处理整个文件夹的图像
        :param input_folder: 输入图像文件夹路径
        :param output_base_folder: 输出基础文件夹路径
        """
        # 设置输出文件夹
        if output_base_folder is None:
            output_base_folder = os.path.join(os.path.dirname(input_folder), "yolov11_results")

        # 创建输出子文件夹
        images_output_folder = os.path.join(output_base_folder, "images_with_boxes")
        labels_output_folder = os.path.join(output_base_folder, "labels")
        os.makedirs(images_output_folder, exist_ok=True)
        os.makedirs(labels_output_folder, exist_ok=True)

        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

        # 获取所有图像文件
        image_files = []
        for filename in os.listdir(input_folder):
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in image_extensions:
                image_files.append(os.path.join(input_folder, filename))

        if not image_files:
            print(f"❌ 在文件夹 {input_folder} 中未找到支持的图像文件")
            return

        print(f"🔍 找到 {len(image_files)} 张图像，开始批量处理...")
        print(f"📁 带框图像将保存至: {images_output_folder}")
        print(f"📄 标签文件将保存至: {labels_output_folder}")

        total_detections = 0
        processed_count = 0

        for i, image_path in enumerate(image_files):
            try:
                # 处理单张图像
                detections = self.process_single_image(
                    image_path, images_output_folder, labels_output_folder
                )
                total_detections += detections
                processed_count += 1

                # 每处理10张图像输出进度
                if (i + 1) % 10 == 0 or (i + 1) == len(image_files):
                    print(f"📊 进度: {i + 1}/{len(image_files)} - 最新检测: {detections}个目标")

            except Exception as e:
                print(f"❌ 处理图像 {os.path.basename(image_path)} 时出错: {e}")
                continue

        print(f"\n✅ 处理完成!")
        print(f"📈 成功处理: {processed_count}/{len(image_files)} 张图像")
        print(f"🔍 总共检测到: {total_detections} 个目标")
        print(f"🖼️  带框图像保存在: {images_output_folder}")
        print(f"📝 标签文件保存在: {labels_output_folder}")

    def process_single_image(self, image_path, images_output_folder, labels_output_folder):
        """
        处理单张图像并保存结果
        :return: 检测到的目标数量
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        orig_height, orig_width = image.shape[:2]
        filename = os.path.basename(image_path)
        name_without_ext = os.path.splitext(filename)[0]

        # 执行推理
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=640,
            verbose=False
        )

        result = results[0]
        boxes = result.boxes

        # 保存带检测框的图像
        annotated_image = result.plot()  # 自动绘制检测框
        output_image_path = os.path.join(images_output_folder, f"{name_without_ext}_detected.jpg")
        cv2.imwrite(output_image_path, annotated_image)

        # 生成YOLO格式的标签内容
        yolo_labels = []
        detection_count = 0

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                # 获取检测信息
                class_id = int(box.cls[0])
                xywhn = box.xywhn[0].cpu().numpy()  # 归一化坐标 [x_center, y_center, width, height]
                confidence = float(box.conf[0])

                # YOLO格式: class_id x_center y_center width height
                label_line = f"{class_id} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}"
                yolo_labels.append(label_line)
                detection_count += 1

        # 保存标签文件
        label_file_path = os.path.join(labels_output_folder, f"{name_without_ext}.txt")
        with open(label_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_labels))

        return detection_count

    def print_detection_summary(self, results):
        """打印检测结果摘要"""
        print("\n📊 检测结果统计:")
        print("-" * 50)

        class_counts = {}
        total_detections = 0

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    total_detections += 1

        # 按数量排序输出
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {count}个")

        print(f"总计: {total_detections}个目标")
        print("-" * 50)


def main():
    """主函数示例"""
    # 配置参数
    model_path = "yolo11n.pt"  # 替换为你的模型路径
    input_folder = "path/to/your/images"  # 替换为你的图像文件夹路径
    output_folder = "detection_results11"  # 结果输出文件夹

    # 创建推理器实例
    detector = YOLOv11FolderInference(
        model_path=model_path,
        conf_threshold=0.5,  # 置信度阈值，可调整
        iou_threshold=0.05  # IOU阈值，可调整
    )

    # 执行文件夹批量推理
    detector.process_folder(input_folder, output_folder)


# 简单使用示例
if __name__ == "__main__":
    # 直接运行示例
    model_path = r"E:\data\前排温度传感器\best.pt"  # 修改为实际模型路径
    image_folder = r"E:\data\2025年12月9日存图"  # 修改为实际图像文件夹路径
    # \\10.20.100.100\datasets\013.nzc\极狐\N50\门把手\ng\门把手
    # \\10.20.100.100\code\013.nzc\极狐\N50\门把手251110\weights
    # 创建推理器
    detector = YOLOv11FolderInference(model_path)

    # 处理整个文件夹
    detector.process_folder(image_folder)

    print("🎉 所有处理完成！")