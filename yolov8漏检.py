from operator import truediv

import cv2
import numpy as np
import onnxruntime as ort
import time
import os
import sys
from pathlib import Path
import json
import shutil

class YOLOv8ONNXInference:
    def __init__(self, onnx_model_path, conf_thres=0.25, iou_thres=0.45):
        """
        初始化YOLOv8 ONNX推理器

        参数:
        onnx_model_path: ONNX模型文件路径
        conf_thres: 置信度阈值
        iou_thres: IOU阈值(NMS)
        """
        # 加载ONNX模型
        self.session = ort.InferenceSession(onnx_model_path,
                                            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        # 获取模型输入信息
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height, self.input_width = self.input_shape[2], self.input_shape[3]
        self.c = 1
        # 获取模型输出信息
        self.output_names = [output.name for output in self.session.get_outputs()]

        # 设置阈值
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # 加载类别名称 (如果有)
        self.class_names = self.load_class_names()

    def load_class_names(self, class_file=None):
        """加载类别名称"""
        if class_file and Path(class_file).exists():
            with open(class_file, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f.readlines()]
        return None

    def preprocess(self, image):
        """
        预处理图像

        参数:
        image: 输入图像 (BGR格式)

        返回:
        preprocessed_image: 预处理后的图像张量
        original_size: 原始图像尺寸 (w, h)
        resized_size: 调整大小后的尺寸 (w, h)
        """
        # 调整大小并保持宽高比
        h, w = image.shape[:2]
        scale = min(self.input_height / h, self.input_width / w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (nw, nh))

        # 创建填充后的图像
        padded = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        padded[:nh, :nw] = resized

        # 转换为RGB并归一化
        padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        padded = padded.astype(np.float32) / 255.0

        # 转换为CHW格式并添加批次维度
        padded = padded.transpose(2, 0, 1)  # HWC -> CHW
        padded = np.expand_dims(padded, axis=0)  # 添加批次维度

        return padded, (w, h), (nw, nh)

    def inference(self, input_tensor):
        """
        执行推理

        参数:
        input_tensor: 预处理后的输入张量

        返回:
        outputs: 模型输出
        """
        return self.session.run(self.output_names, {self.input_name: input_tensor})

    def postprocess(self, outputs, original_size, resized_size):
        """
        后处理模型输出

        参数:
        outputs: 模型输出
        original_size: 原始图像尺寸 (w, h)
        resized_size: 调整大小后的尺寸 (w, h)

        返回:
        detections: 检测结果列表 [x1, y1, x2, y2, conf, class_id]
        """
        # YOLOv8 ONNX模型输出格式: [batch, num_classes+4, num_detections]
        predictions = np.squeeze(outputs[0]).T

        # 过滤低置信度检测
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_thres, :]
        scores = scores[scores > self.conf_thres]

        if len(scores) == 0:
            return []

        # 获取类别ID
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # 提取边界框 (cx, cy, w, h)
        boxes = predictions[:, :4]

        # 将边界框转换为 (x1, y1, x2, y2)
        boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1 = cx - w/2
        boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1 = cy - h/2
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2 = x1 + w
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2 = y1 + h

        # 应用非极大值抑制 (NMS)
        indices = self.non_max_suppression(boxes, scores, self.iou_thres)

        # 提取最终检测结果
        detections = []
        for i in indices:
            x1, y1, x2, y2 = boxes[i]
            conf = scores[i]
            class_id = class_ids[i]

            # 缩放回原始图像尺寸
            scale_w = original_size[0] / resized_size[0]
            scale_h = original_size[1] / resized_size[1]

            x1 = int(x1 * scale_w)
            y1 = int(y1 * scale_h)
            x2 = int(x2 * scale_w)
            y2 = int(y2 * scale_h)

            detections.append([x1, y1, x2, y2, conf, class_id])

        return detections

    def non_max_suppression(self, boxes, scores, iou_threshold):
        """
        非极大值抑制 (NMS)

        参数:
        boxes: 边界框 [N, 4] (x1, y1, x2, y2)
        scores: 置信度分数 [N]
        iou_threshold: IOU阈值

        返回:
        indices: 保留的检测索引
        """
        # 按分数降序排序
        sorted_indices = np.argsort(scores)[::-1]

        keep_indices = []
        while sorted_indices.size > 0:
            # 获取当前最高分数框
            i = sorted_indices[0]
            keep_indices.append(i)

            # 计算当前框与其他框的IOU
            current_box = boxes[i]
            other_boxes = boxes[sorted_indices[1:]]

            # 计算IOU
            ious = self.calculate_iou(current_box, other_boxes)

            # 移除IOU大于阈值的框
            overlapping_indices = np.where(ious > iou_threshold)[0]
            sorted_indices = np.delete(sorted_indices, [0] + list(overlapping_indices + 1))

        return keep_indices

    def calculate_iou(self, box, boxes):
        """
        计算一个框与多个框的IOU

        参数:
        box: 单个框 [x1, y1, x2, y2]
        boxes: 多个框 [N, 4] (x1, y1, x2, y2)

        返回:
        ious: IOU值数组 [N]
        """
        # 计算交集区域
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        # 计算交集面积
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        # 计算并集面积
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area_box + area_boxes - intersection

        # 计算IOU
        iou = intersection / (union + 1e-6)
        return iou

    def draw_detections(self, image, detections):
        """
        在图像上绘制检测结果

        参数:
        image: 原始图像
        detections: 检测结果列表 [x1, y1, x2, y2, conf, class_id]

        返回:
        image_with_detections: 绘制了检测结果的图像
        """
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det

            # 绘制边界框
            color = (0, 255, 0)  # 绿色
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # 创建标签文本
            if self.class_names:
                class_name = self.class_names[int(class_id)]
            else:
                class_name = f"Class {int(class_id)}"

            label = f"{class_name}: {conf:.2f}"

            # 计算文本位置
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # 绘制文本背景
            cv2.rectangle(image, (x1, y1 - text_height - 5),
                          (x1 + text_width, y1), color, -1)

            # 绘制文本
            cv2.putText(image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return image

    def safe_read_image(self, image_path):
        """
        安全读取图像（处理中文路径）

        参数:
        image_path: 图像文件路径（可能包含中文）

        返回:
        image: 读取的图像，如果失败返回None
        """
        try:
            # 方法1: 使用OpenCV的imread（可能不支持中文路径）
            image = cv2.imread(image_path)
            if image is not None:
                return image

            # 方法2: 使用numpy和OpenCV的imdecode（支持中文路径）
            with open(image_path, 'rb') as f:
                image_data = np.frombuffer(f.read(), np.uint8)
                image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                return image
        except Exception as e:
            print(f"读取图像失败: {image_path}, 错误: {e}")
            return None

    def safe_write_image(self, image, output_path):
        """
        安全保存图像（处理中文路径）

        参数:
        image: 要保存的图像
        output_path: 输出文件路径（可能包含中文）

        返回:
        success: 是否成功保存
        """
        try:
            # 方法1: 使用OpenCV的imwrite（可能不支持中文路径）
            # success = cv2.imwrite(output_path, image)
            # if success:
            #     return True

            # 方法2: 使用OpenCV的imencode（支持中文路径）
            success, encoded_image = cv2.imencode('.jpg', image)
            if success:
                with open(output_path, 'wb') as f:
                    f.write(encoded_image.tobytes())
                return True
            return False
        except Exception as e:
            print(f"保存图像失败: {output_path}, 错误: {e}")
            return False

    def process_folder(self, input_folder, output_folder, save_txt=False, save_json=False):
        """
        处理整个文件夹中的图像（支持中文路径）

        参数:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        save_txt: 是否保存检测结果为TXT文件
        save_json: 是否保存检测结果为JSON文件
        """
        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)

        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

        # 获取所有图像文件
        image_files = []
        for file in os.listdir(input_folder):
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(file)

        if not image_files:
            print(f"在文件夹 {input_folder} 中未找到图像文件")
            return

        print(f"找到 {len(image_files)} 个图像文件")

        # 处理统计
        total_time = 0
        total_detections = 0

        # 处理每个图像
        for i, image_file in enumerate(image_files):
            print(f"处理图像 {i + 1}/{len(image_files)}: {image_file}")

            # 构建完整路径（使用Unicode字符串）
            image_path = os.path.join(input_folder, image_file)

            # 处理单个图像
            self.process_single_image(image_path, output_folder, image_file)

            # # 更新统计
            # total_time += process_time
            # total_detections += len(detections)

            # # 保存检测结果
            # if save_txt:
            #     self.save_detections_txt(detections, output_folder, image_file)
            #
            # if save_json:
            #     self.save_detections_json(detections, output_folder, image_file)

        # 打印统计信息
        print(f"\n处理完成!")
        print(f"总处理时间: {total_time:.2f}秒")
        print(f"平均每张图像处理时间: {total_time / len(image_files):.2f}秒")
        print(f"检测到的对象总数: {total_detections}")
        print(f"平均每张图像检测对象数: {total_detections / len(image_files):.2f}")
        print(f"结果保存在: {output_folder}")

    def process_single_image(self, image_path, output_folder, image_file):
        """
        处理单个图像（支持中文路径）

        参数:
        image_path: 图像完整路径
        output_folder: 输出文件夹路径
        image_file: 图像文件名

        返回:
        detections: 检测结果
        process_time: 处理时间
        """
        start_time = time.time()

        # 安全加载图像（支持中文路径）
        image = self.safe_read_image(image_path)
        if image is None:
            print(f"错误: 无法加载图像 {image_path}")
            return [], 0  # 确保返回元组 ([], 0)

        # 预处理
        input_tensor, original_size, resized_size = self.preprocess(image)

        # 推理
        outputs = self.inference(input_tensor)

        # 后处理
        detections = self.postprocess(outputs, original_size, resized_size)
        # if detections[0][5]!=0:
        #     print(1111)
        # 绘制检测结果
        result_image = self.draw_detections(image.copy(), detections)

        # 安全保存结果图像（支持中文路径）
        output_path = os.path.join(output_folder, f"detected_{image_file}")
        success = self.safe_write_image(result_image, output_path)



        process_time = time.time() - start_time

        print(f"  - 检测到 {len(detections)} 个对象, 处理时间: {process_time:.2f}秒")


        destination_folder = output_folder + "\\loujian\\"
        os.makedirs(destination_folder, exist_ok=True)
        if len(detections)==0:
            self.c+=1
            image_file_ = str(self.c)+image_file
            destination_path = os.path.join(destination_folder, image_file_)

            try:
                # 复制图像文件
                shutil.copy2(image_path, destination_path)
                print(f"成功复制: {image_path} -> {destination_path}")
                return True
            except Exception as e:
                print(f"复制失败: {image_path}, 错误: {e}")
                # 确保返回元组
        return detections, process_time

    def save_detections_txt(self, detections, output_folder, image_file):
        """
        保存检测结果为TXT文件 (YOLO格式)

        参数:
        detections: 检测结果
        output_folder: 输出文件夹路径
        image_file: 图像文件名
        """
        txt_filename = Path(image_file).stem + ".txt"
        txt_path = os.path.join(output_folder, txt_filename)

        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                for det in detections:
                    x1, y1, x2, y2, conf, class_id = det
                    # YOLO格式: class_id x_center y_center width height confidence
                    width = x2 - x1
                    height = y2 - y1
                    x_center = x1 + width / 2
                    y_center = y1 + height / 2
                    # 归一化坐标 (假设图像尺寸已记录)
                    # 这里需要知道图像尺寸，但为了简化，我们保存绝对坐标
                    f.write(f"{int(class_id)} {x_center} {y_center} {width} {height} {conf:.4f}\n")
        except Exception as e:
            print(f"保存TXT文件失败: {txt_path}, 错误: {e}")

    def save_detections_json(self, detections, output_folder, image_file):
        """
        保存检测结果为JSON文件

        参数:
        detections: 检测结果
        output_folder: 输出文件夹路径
        image_file: 图像文件名
        """
        json_filename = Path(image_file).stem + ".json"
        json_path = os.path.join(output_folder, json_filename)

        try:
            # 转换检测结果为字典格式
            detections_dict = []
            for i, det in enumerate(detections):
                x1, y1, x2, y2, conf, class_id = det
                detection_info = {
                    "id": i,
                    "class_id": int(class_id),
                    "class_name": self.class_names[int(class_id)] if self.class_names else f"Class {int(class_id)}",
                    "confidence": float(conf),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                }
                detections_dict.append(detection_info)

            # 保存为JSON
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(detections_dict, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存JSON文件失败: {json_path}, 错误: {e}")


def main():
    # 配置参数
    onnx_model_path = r"E:\data\前排温度传感器\best.onnx"  # 替换为您的ONNX模型路径
    input_folder = r"E:\data\2025年12月9日存图"  # 替换为您的图像文件夹路径（可以包含中文）
    output_folder = r"E:\data\2025年12月9日存图—result"  # 输出文件夹路径
    # class_names_path = "coco.names"  # 替换为您的类别名称文件路径 (可选)#

    # 设置Python环境以支持中文
    if sys.platform.startswith('win'):
        # Windows系统设置
        import locale
        locale.setlocale(locale.LC_ALL, 'chinese')

    # 创建推理器
    detector = YOLOv8ONNXInference(onnx_model_path, conf_thres=0.6, iou_thres=0.45)

    # # 加载类别名称 (可选)
    # if class_names_path and Path(class_names_path).exists():
    #     detector.class_names = detector.load_class_names(class_names_path)

    # 处理整个文件夹
    detector.process_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        save_txt=True,  # 是否保存TXT格式的检测结果
        save_json=False  # 是否保存JSON格式的检测结果
    )


if __name__ == "__main__":
    main()