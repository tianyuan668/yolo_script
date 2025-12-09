import cv2
import numpy as np
import onnxruntime as ort
import time
import os
import sys
from pathlib import Path
import json
import shutil


class YOLOv5ONNXInference:
    def __init__(self, onnx_model_path, conf_thres=0.25, iou_thres=0.45):
        """
        初始化YOLOv5 ONNX推理器

        参数:
        onnx_model_path: ONNX模型文件路径
        conf_thres: 置信度阈值
        iou_thres: IOU阈值(NMS)
        """
        # 加载ONNX模型
        self.session = ort.InferenceSession(onnx_model_path)

        # 获取模型输入信息
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height, self.input_width = self.input_shape[2], self.input_shape[3]

        # 设置阈值
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # 加载COCO类别名称
        self.class_names = self.load_coco_class_names()

        # 计数器（用于未检测到对象的图像）
        self.counter = 0

    def load_coco_class_names(self):
        """加载COCO数据集类别名称"""
        return ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
                'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def preprocess(self, image):
        """
        预处理图像 - 保持宽高比的填充调整[3,4](@ref)

        参数:
        image: 输入图像 (BGR格式)

        返回:
        preprocessed_image: 预处理后的图像张量
        ratio: 缩放比例
        pad: 填充尺寸
        """
        # 原始图像尺寸
        h, w = image.shape[:2]

        # 计算缩放比例并调整大小
        scale = min(self.input_height / h, self.input_width / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # 调整图像大小
        resized = cv2.resize(image, (new_w, new_h))

        # 创建填充后的图像
        padded = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        pad_h = (self.input_height - new_h) // 2
        pad_w = (self.input_width - new_w) // 2
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # 转换为RGB并归一化
        padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        padded = padded.astype(np.float32) / 255.0

        # 转换为CHW格式并添加批次维度
        padded = padded.transpose(2, 0, 1)  # HWC -> CHW
        padded = np.expand_dims(padded, axis=0)  # 添加批次维度

        return padded, scale, (pad_w, pad_h)

    def inference(self, input_tensor):
        """
        执行推理[1,4](@ref)

        参数:
        input_tensor: 预处理后的输入张量

        返回:
        outputs: 模型输出
        """
        return self.session.run(None, {self.input_name: input_tensor})[0]

    def postprocess(self, outputs, original_shape, scale, pad):
        """
        后处理模型输出 - 修复坐标转换问题[2,5](@ref)

        参数:
        outputs: 模型输出
        original_shape: 原始图像尺寸 (h, w)
        scale: 缩放比例
        pad: 填充尺寸 (pad_w, pad_h)

        返回:
        detections: 检测结果列表 [x1, y1, x2, y2, conf, class_id]
        """
        # 压缩输出维度
        predictions = np.squeeze(outputs)  # [num_boxes, 85]

        # 过滤低置信度检测
        conf_mask = predictions[:, 4] > self.conf_thres
        predictions = predictions[conf_mask]

        if len(predictions) == 0:
            return []

        # 获取类别分数和ID
        class_scores = predictions[:, 5:]
        class_ids = np.argmax(class_scores, axis=1)
        max_class_scores = np.max(class_scores, axis=1)

        # 综合置信度
        final_scores = predictions[:, 4] * max_class_scores

        # 提取边界框 (x_center, y_center, width, height)
        boxes = predictions[:, :4].copy()

        # 转换为xyxy格式
        boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2

        # 应用NMS
        indices = self.non_max_suppression(boxes, final_scores, self.iou_thres)

        # 处理检测结果
        detections = []
        for i in indices:
            x1, y1, x2, y2 = boxes[i]
            conf = final_scores[i]
            class_id = class_ids[i]

            # 去除填充并缩放到原始尺寸
            pad_w, pad_h = pad
            x1 = max(0, (x1 - pad_w) / scale)
            y1 = max(0, (y1 - pad_h) / scale)
            x2 = min(original_shape[1], (x2 - pad_w) / scale)
            y2 = min(original_shape[0], (y2 - pad_h) / scale)

            # 确保坐标在有效范围内
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            detections.append([x1, y1, x2, y2, conf, class_id])

        return detections

    def non_max_suppression(self, boxes, scores, iou_threshold):
        """非极大值抑制[2](@ref)"""
        if len(boxes) == 0:
            return []

        # 按分数降序排序
        sorted_indices = np.argsort(scores)[::-1]

        keep_indices = []
        while sorted_indices.size > 0:
            # 取当前最高分框
            i = sorted_indices[0]
            keep_indices.append(i)

            if sorted_indices.size == 1:
                break

            # 计算IOU
            current_box = boxes[i]
            other_boxes = boxes[sorted_indices[1:]]

            ious = self.calculate_iou(current_box, other_boxes)

            # 保留IOU低于阈值的框
            overlapping_indices = np.where(ious <= iou_threshold)[0]
            sorted_indices = sorted_indices[overlapping_indices + 1]

        return keep_indices

    def calculate_iou(self, box, boxes):
        """计算IOU[2](@ref)"""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area_box + area_boxes - intersection

        return intersection / (union + 1e-6)

    def save_detections_txt(self, detections, output_path, original_shape):
        """
        保存检测结果为YOLO格式TXT文件 - 修复格式问题[6](@ref)

        参数:
        detections: 检测结果
        output_path: 输出文件路径
        original_shape: 原始图像尺寸 (h, w)
        """
        if not detections:
            return  # 无检测结果不保存文件

        original_h, original_w = original_shape

        with open(output_path, 'w', encoding='utf-8') as f:
            for det in detections:
                x1, y1, x2, y2, conf, class_id = det

                # 转换为YOLO格式：class x_center y_center width height
                width = x2 - x1
                height = y2 - y1
                x_center = x1 + width / 2
                y_center = y1 + height / 2

                # 归一化坐标
                x_center_norm = x_center / original_w
                y_center_norm = y_center / original_h
                width_norm = width / original_w
                height_norm = height / original_h

                # 写入文件：class x_center y_center width height confidence
                line = f"{int(class_id)} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n"
                f.write(line)

    def draw_detections(self, image, detections):
        """在图像上绘制检测结果[3](@ref)"""
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det

            # 绘制边界框
            color = (0, 255, 0)  # 绿色
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # 创建标签
            class_name = self.class_names[int(class_id)] if self.class_names else f"Class {int(class_id)}"
            label = f"{class_name} {conf:.2f}"

            # 计算文本尺寸
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # 绘制文本背景
            cv2.rectangle(image, (x1, y1 - text_height - 5),
                          (x1 + text_width, y1), color, -1)

            # 绘制文本
            cv2.putText(image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return image

    def process_single_image(self, image_path, output_dir):
        """
        处理单张图像

        参数:
        image_path: 输入图像路径
        output_dir: 输出目录

        返回:
        detections: 检测结果
        process_time: 处理时间
        """
        start_time = time.time()

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"错误: 无法读取图像 {image_path}")
            return [], 0

        original_h, original_w = image.shape[:2]

        try:
            # 预处理
            input_tensor, scale, pad = self.preprocess(image)

            # 推理
            outputs = self.inference(input_tensor)

            # 后处理
            detections = self.postprocess(outputs, (original_h, original_w), scale, pad)

            # 绘制检测结果
            result_image = self.draw_detections(image.copy(), detections)

            # 保存结果图像
            output_image_path = os.path.join(output_dir, f"detected_{Path(image_path).name}")
            cv2.imwrite(output_image_path, result_image)

            # 保存TXT结果
            txt_path = os.path.join(output_dir, f"{Path(image_path).stem}.txt")
            self.save_detections_txt(detections, txt_path, (original_h, original_w))

            process_time = time.time() - start_time

            print(f"处理完成: {Path(image_path).name}")
            print(f"检测到 {len(detections)} 个对象, 耗时: {process_time:.2f}秒")

            # 如果没有检测到对象，复制到特殊文件夹
            if len(detections) == 0:
                self.copy_no_detection_image(image_path, output_dir)

            return detections, process_time

        except Exception as e:
            print(f"处理图像时出错 {image_path}: {e}")
            return [], 0

    def copy_no_detection_image(self, image_path, output_dir):
        """复制未检测到对象的图像到指定文件夹"""
        no_detection_dir = os.path.join(output_dir, "no_detections")
        os.makedirs(no_detection_dir, exist_ok=True)

        self.counter += 1
        dest_path = os.path.join(no_detection_dir, f"{self.counter}_{Path(image_path).name}")

        try:
            shutil.copy2(image_path, dest_path)
            print(f"复制未检测图像到: {dest_path}")
        except Exception as e:
            print(f"复制失败: {e}")

    def process_folder(self, input_folder, output_folder):
        """
        处理整个文件夹

        参数:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        """
        # 创建输出目录
        os.makedirs(output_folder, exist_ok=True)

        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

        # 获取所有图像文件
        image_files = []
        for file in os.listdir(input_folder):
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(file)

        if not image_files:
            print(f"在 {input_folder} 中未找到图像文件")
            return

        print(f"找到 {len(image_files)} 个图像文件")

        # 处理统计
        total_time = 0
        total_detections = 0

        # 处理每个图像
        for i, image_file in enumerate(image_files, 1):
            image_path = os.path.join(input_folder, image_file)
            print(f"\n处理进度: {i}/{len(image_files)} - {image_file}")

            detections, process_time = self.process_single_image(image_path, output_folder)

            total_time += process_time
            total_detections += len(detections)

        # 打印统计信息
        print(f"\n处理完成!")
        print(f"总处理时间: {total_time:.2f}秒")
        print(f"平均每张图像处理时间: {total_time / len(image_files):.2f}秒")
        print(f"检测到的对象总数: {total_detections}")
        print(f"结果保存在: {output_folder}")


def main():
    """主函数"""
    # 配置参数
    onnx_model_path = r"D:\test_images\948W\12-8\best.onnx"  # 替换为你的ONNX模型路径
    input_folder = r"D:\test_images\948W\12-8\12-8"  # 输入图像文件夹
    output_folder = r"D:\test_images\948W\12-8\12-8\results"  # 输出文件夹

    # 创建推理器
    detector = YOLOv5ONNXInference(onnx_model_path)

    # 处理文件夹
    detector.process_folder(input_folder, output_folder)

if __name__ == "__main__":
    main()