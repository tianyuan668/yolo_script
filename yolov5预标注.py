import cv2
import numpy as np
import onnxruntime as ort
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import time


class YOLOv5ONNXInference:
    def __init__(self, model_path, conf_thres=0.25, iou_thres=0.45):
        """
        初始化YOLOv5 ONNX推理器

        Args:
            model_path: ONNX模型路径
            conf_thres: 置信度阈值
            iou_thres: IOU阈值用于NMS
        """
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # 加载ONNX模型
        self.session = ort.InferenceSession(model_path,
                                            providers=['CUDAExecutionProvider',
                                                       'CPUExecutionProvider'])

        # 获取模型输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # 获取模型输入尺寸
        self.input_shape = self.session.get_inputs()[0].shape
        self.model_height = self.input_shape[2]
        self.model_width = self.input_shape[3]

        print(f"模型输入尺寸: {self.model_width}x{self.model_height}")

    def preprocess(self, image):
        """
        图像预处理

        Args:
            image: 输入图像 (H,W,C) BGR格式

        Returns:
            blob: 预处理后的blob
            original_shape: 原始图像尺寸
            ratio: 缩放比例
            pad: 填充信息
        """
        original_shape = image.shape[:2]  # (H, W)

        # 计算缩放比例，保持宽高比
        r = min(self.model_height / original_shape[0],
                self.model_width / original_shape[1])

        # 计算新的尺寸
        new_size = (int(original_shape[1] * r), int(original_shape[0] * r))

        # 缩放图像
        resized = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

        # 创建画布并填充到模型输入尺寸
        canvas = np.full((self.model_height, self.model_width, 3), 114, dtype=np.uint8)

        # 计算填充位置
        top = (self.model_height - new_size[1]) // 2
        left = (self.model_width - new_size[0]) // 2

        # 将缩放后的图像放到画布中心
        canvas[top:top + new_size[1], left:left + new_size[0]] = resized

        # BGR转RGB
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        # 归一化 (0-1)
        blob = canvas.astype(np.float32) / 255.0

        # 调整维度顺序为 NCHW
        blob = np.transpose(blob, (2, 0, 1))  # HWC -> CHW
        blob = np.expand_dims(blob, 0)  # CHW -> NCHW

        # 保存转换信息用于后处理
        info = {
            'original_shape': original_shape,
            'ratio': r,
            'pad': (top, left)
        }

        return blob, info

    def postprocess(self, outputs, info):
        """
        后处理：解析模型输出

        Args:
            outputs: 模型输出
            info: 预处理信息

        Returns:
            detections: 检测结果列表，每个元素为 [x1, y1, x2, y2, conf, class_id]
        """
        detections = []
        original_shape = info['original_shape']
        ratio = info['ratio']
        pad_top, pad_left = info['pad']

        # 模型输出形状通常是 (1, 25200, 85) 或其他，取决于模型配置
        # 其中85 = 4(xywh) + 1(conf) + 80(classes)
        outputs = outputs[0]  # 去掉batch维度

        # 遍历所有预测
        for pred in outputs:
            # 提取边界框坐标和置信度
            x_center, y_center, width, height = pred[0:4]
            conf = pred[4]

            # 只处理置信度高于阈值的预测
            if conf < self.conf_thres:
                continue

            # 提取类别分数
            class_scores = pred[5:]
            class_id = np.argmax(class_scores)
            class_conf = class_scores[class_id]

            # 最终置信度 = 物体置信度 * 类别置信度
            final_conf = conf * class_conf

            if final_conf < self.conf_thres:
                continue

            # 将中心坐标转换为左上角和右下角坐标
            x1 = (x_center - width / 2)
            y1 = (y_center - height / 2)
            x2 = (x_center + width / 2)
            y2 = (y_center + height / 2)

            # 从模型输入尺寸映射回原始图像尺寸
            # 1. 减去填充
            x1 = max(0, x1 - pad_left)
            y1 = max(0, y1 - pad_top)
            x2 = max(0, x2 - pad_left)
            y2 = max(0, y2 - pad_top)

            # 2. 除以缩放比例
            x1 /= ratio
            y1 /= ratio
            x2 /= ratio
            y2 /= ratio

            # 确保坐标在图像范围内
            x1 = max(0, min(x1, original_shape[1]))
            y1 = max(0, min(y1, original_shape[0]))
            x2 = max(0, min(x2, original_shape[1]))
            y2 = max(0, min(y2, original_shape[0]))

            # 添加到检测结果
            detections.append([x1, y1, x2, y2, final_conf, class_id])

        # 应用NMS (非极大值抑制)
        detections = self.non_max_suppression(detections)

        return detections

    def non_max_suppression(self, detections):
        """
        非极大值抑制

        Args:
            detections: 检测结果列表

        Returns:
            经过NMS过滤的检测结果
        """
        if len(detections) == 0:
            return []

        # 将列表转换为numpy数组
        boxes = np.array([det[:4] for det in detections])
        scores = np.array([det[4] for det in detections])
        class_ids = np.array([det[5] for det in detections])

        # 计算每个框的面积
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # 按置信度降序排序
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            # 取置信度最高的框
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break

            # 计算当前框与其他框的IOU
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # 保留IOU低于阈值的框
            inds = np.where(iou <= self.iou_thres)[0]
            order = order[inds + 1]

        # 返回过滤后的结果
        return [detections[i] for i in keep]

    def convert_to_yolo_format(self, detections, image_shape):
        """
        将检测结果转换为YOLO标注格式

        Args:
            detections: 检测结果列表
            image_shape: 图像尺寸 (H, W)

        Returns:
            yolo_labels: YOLO格式标注字符串列表
        """
        yolo_labels = []
        img_height, img_width = image_shape

        for det in detections:
            x1, y1, x2, y2, conf, class_id = det

            # 转换为YOLO格式: class_id x_center y_center width height
            # 坐标归一化到 [0, 1]
            x_center = (x1 + x2) / 2 / img_width
            y_center = (y1 + y2) / 2 / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height

            # YOLO格式通常不包含置信度，但可以添加
            # 这里我们按照标准YOLO格式，只保存类别和归一化坐标
            label = f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            yolo_labels.append(label)

        return yolo_labels

    def detect_image(self, image_path):
        """
        检测单张图像

        Args:
            image_path: 图像路径

        Returns:
            detections: 检测结果
            original_image: 原始图像
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return None, None

        # 预处理
        blob, info = self.preprocess(image)

        # 推理
        start_time = time.time()
        outputs = self.session.run([self.output_name], {self.input_name: blob})
        inference_time = time.time() - start_time

        # 后处理
        detections = self.postprocess(outputs, info)

        print(f"图像: {Path(image_path).name}, 推理时间: {inference_time:.3f}s, 检测到 {len(detections)} 个对象")

        return detections, image

    def save_yolo_labels(self, detections, image_shape, output_path):
        """
        保存YOLO格式标注

        Args:
            detections: 检测结果
            image_shape: 图像尺寸
            output_path: 输出文件路径
        """
        # 转换为YOLO格式
        yolo_labels = self.convert_to_yolo_format(detections, image_shape)

        # 保存到文件
        with open(output_path, 'w') as f:
            for label in yolo_labels:
                f.write(label + '\n')

    def visualize_results(self, image, detections, output_path=None):
        """
        可视化检测结果（可选）

        Args:
            image: 原始图像
            detections: 检测结果
            output_path: 可视化结果保存路径
        """
        if detections is None or len(detections) == 0:
            return image

        # 定义颜色（可以根据类别数量调整）
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)
        ]

        result_image = image.copy()

        for det in detections:
            x1, y1, x2, y2, conf, class_id = det

            # 转换为整数坐标
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 选择颜色
            color = colors[class_id % len(colors)]

            # 画边界框
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

            # 标签文本
            label = f"Class {class_id}: {conf:.2f}"

            # 计算文本位置
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # 画文本背景
            cv2.rectangle(result_image, (x1, y1 - text_height - 4),
                          (x1 + text_width, y1), color, -1)

            # 画文本
            cv2.putText(result_image, label, (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 保存可视化结果
        if output_path:
            cv2.imwrite(output_path, result_image)

        return result_image


def process_folder(args):
    """
    处理整个文件夹
    """
    # 初始化推理器
    detector = YOLOv5ONNXInference(
        model_path=args.model,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres
    )

    # 获取图像文件列表
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []

    for ext in image_extensions:
        image_files.extend(Path(args.input_folder).glob(f"*{ext}"))
        image_files.extend(Path(args.input_folder).glob(f"*{ext.upper()}"))

    print(f"找到 {len(image_files)} 张图像")

    # 创建输出文件夹
    labels_dir = Path(args.output_folder) / "labels"
    images_dir = Path(args.output_folder) / "images"
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # 如果需要可视化，创建可视化文件夹
    if args.visualize:
        viz_dir = Path(args.output_folder) / "visualization"
        viz_dir.mkdir(parents=True, exist_ok=True)

    # 处理每张图像
    for image_path in tqdm(image_files, desc="处理图像"):
        # 检测
        detections, image = detector.detect_image(str(image_path))

        if detections is None:
            continue

        # 保存YOLO格式标注
        label_filename = image_path.stem + ".txt"
        label_path = labels_dir / label_filename
        detector.save_yolo_labels(detections, image.shape[:2], str(label_path))

        # 如果需要，保存可视化结果
        if args.visualize:
            viz_filename = image_path.stem + "_detected.jpg"
            viz_path = viz_dir / viz_filename
            detector.visualize_results(image, detections, str(viz_path))

        # 如果需要，复制原始图像到输出文件夹
        if args.save_images:
            output_image_path = images_dir / image_path.name
            cv2.imwrite(str(output_image_path), image)

    print(f"处理完成！结果保存在: {args.output_folder}")


def main():
    parser = argparse.ArgumentParser(description="YOLOv5 ONNX模型推理")
    parser.add_argument("--model", type=str, required=True, help="ONNX模型路径")
    parser.add_argument("--input-folder", type=str, required=True, help="输入图像文件夹路径")
    parser.add_argument("--output-folder", type=str, default="./results", help="输出文件夹路径")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IOU阈值")
    parser.add_argument("--visualize", action="store_true", help="是否生成可视化结果")
    parser.add_argument("--save-images", action="store_true", help="是否保存原始图像到输出文件夹")

    args = parser.parse_args()

    # 处理文件夹
    process_folder(args)


if __name__ == "__main__":
    main()