import os
import cv2
import numpy as np
import onnxruntime as ort


class YOLOv11Detector:
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_shape = self.session.get_inputs()[0].shape[2:]  # (H, W)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        print(f"✅ 模型加载成功: {model_path} (输入尺寸:{self.input_shape[1]}x{self.input_shape[0]})")

    def _preprocess(self, img_path):
        """增强型图像预处理"""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")

        # 保持宽高比缩放
        h, w = img.shape[:2]
        scale = min(self.input_shape[1] / w, self.input_shape[0] / h)
        new_size = (int(w * scale), int(h * scale))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

        # 中心对称填充
        pad_w = (self.input_shape[1] - new_size[0]) // 2
        pad_h = (self.input_shape[0] - new_size[1]) // 2
        padded_img = np.full((*self.input_shape, 3), 114, dtype=np.uint8)
        padded_img[pad_h:pad_h + new_size[1], pad_w:pad_w + new_size[0]] = img

        # 转换为模型输入格式
        img_array = padded_img.astype(np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))  # HWC → CHW
        return np.expand_dims(img_array, axis=0)

    def _postprocess(self, outputs, orig_size):
        """增强型后处理"""
        boxes = []
        scores = []
        class_ids = []

        # 解析所有输出节点
        for output in outputs:
            raw_detections = output.reshape(-1, 85)  # 假设输出为 [x1,y1,x2,y2,conf,class1...class80]

            for detection in raw_detections:
                # 过滤低置信度检测
                if detection[4] < self.conf_threshold:
                    continue

                # 提取坐标和置信度
                x1, y1, x2, y2 = detection[0:4]
                conf = detection[4]

                # 计算类别概率
                class_probs = detection[5:]
                class_id = np.argmax(class_probs)

                # 转换为YOLO格式的归一化坐标
                x_center = (x1 + x2) / 2 / self.input_shape[1]
                y_center = (y1 + y2) / 2 / self.input_shape[0]
                width = (x2 - x1) / self.input_shape[1]
                height = (y2 - y1) / self.input_shape[0]

                boxes.append([x_center, y_center, width, height])
                scores.append(float(conf))
                class_ids.append(int(class_id))

        # 应用非极大值抑制
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)
        return [(class_ids[i], boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])
                for i in indices]

    def detect_folder(self, input_folder, output_folder):
        """批量处理图像文件夹"""
        os.makedirs(output_folder, exist_ok=True)

        for filename in os.listdir(input_folder):
            if not filename.lower().endswith(('jpg', 'jpeg', 'png', 'bmp')):
                continue

            img_path = os.path.join(input_folder, filename)
            try:
                # 执行推理
                input_tensor = self._preprocess(img_path)
                outputs = self.session.run(self.output_names,
                                           {self.input_name: input_tensor})

                # 后处理
                detections = self._postprocess(outputs, orig_size=cv2.imread(img_path).shape[:2])

                # 保存结果
                self._save_results(img_path, detections, output_folder)

            except Exception as e:
                print(f"❌ 处理 {filename} 时出错: {str(e)}")

    def _save_results(self, img_path, detections, output_folder):
        """保存YOLO格式的TXT文件"""
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(output_folder, f"{base_name}.txt")

        with open(txt_path, 'w') as f:
            for det in detections:
                class_id, x_center, y_center, width, height = det
                line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                f.write(line + '\n')

if __name__ == "__main__":
    # 配置参数
    model_path = r"E:\data\zuoqianqieshuitiao\best.onnx"  # 替换为你的ONNX模型路径
    input_folder = r"E:\data\zuoqianqieshuitiao\B"  # 输入图像文件夹
    output_folder = r"E:\data\zuoqianqieshuitiao\output_labels"  # 输出TXT文件夹

    # 初始化检测器
    detector = YOLOv11Detector(
        model_path=model_path,
        conf_threshold=0.5,
        iou_threshold=0.45
    )

    # 执行推理
    detector.detect_folder(input_folder, output_folder)