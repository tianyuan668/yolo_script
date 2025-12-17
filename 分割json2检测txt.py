import json
import os
import glob

def convert_seg_json_to_det_txt(json_file_path, output_txt_path, class_list=None):
    """
    将分割JSON标注文件转换为YOLO目标检测格式的TXT文件
    
    Args:
        json_file_path (str): 输入的JSON文件路径
        output_txt_path (str): 输出的TXT文件路径
        class_list (list, optional): 类别名称列表
    """
    
    try:
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 获取图像尺寸
        image_height = data['imageHeight']
        image_width = data['imageWidth']
        
        # 创建或清空输出TXT文件
        with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
            # 处理每个分割形状
            for shape in data['shapes']:
                label = shape['label']
                points = shape['points']
                shape_type = shape['shape_type']
                
                # 确定类别ID
                if class_list is not None:
                    # 如果提供了类别列表，查找标签对应的索引
                    if label in class_list:
                        class_id = class_list.index(label)
                    else:
                        # 如果标签不在列表中，添加到列表末尾
                        class_list.append(label)
                        class_id = len(class_list) - 1
                        print(f"警告: 发现新类别 '{label}'，已分配ID: {class_id}")
                else:
                    # 如果没有提供类别列表，直接使用标签作为ID
                    try:
                        class_id = int(label)
                    except ValueError:
                        # 如果标签不是数字，使用0作为默认ID
                        class_id = 0
                        print(f"警告: 标签 '{label}' 不是数字，使用默认ID: 0")
                
                # 从多边形点计算边界框 [1](@ref)
                x_coords = [point[0] for point in points]
                y_coords = [point[1] for point in points]
                
                x_min = min(x_coords)
                x_max = max(x_coords)
                y_min = min(y_coords)
                y_max = max(y_coords)
                
                # 计算边界框的中心点、宽度和高度
                bbox_width = x_max - x_min
                bbox_height = y_max - y_min
                x_center = x_min + bbox_width / 2
                y_center = y_min + bbox_height / 2
                
                # 归一化到 [0, 1] 范围 [7](@ref)
                x_center_norm = x_center / image_width
                y_center_norm = y_center / image_height
                width_norm = bbox_width / image_width
                height_norm = bbox_height / image_height
                
                # 确保坐标在有效范围内
                x_center_norm = max(0.0, min(1.0, x_center_norm))
                y_center_norm = max(0.0, min(1.0, y_center_norm))
                width_norm = max(0.0, min(1.0, width_norm))
                height_norm = max(0.0, min(1.0, height_norm))
                
                # 写入TXT文件：YOLO目标检测格式 [7](@ref)
                # 格式: <class_id> <x_center> <y_center> <width> <height>
                txt_file.write(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")
        
        print(f"成功转换: {json_file_path} -> {output_txt_path}")
        
    except Exception as e:
        print(f"转换失败 {json_file_path}: {str(e)}")

def batch_convert_seg_json_to_det_txt(json_folder_path, output_folder_path, class_list=None):
    """
    批量转换文件夹中的所有分割JSON文件为YOLO目标检测TXT格式
    
    Args:
        json_folder_path (str): 包含JSON文件的文件夹路径
        output_folder_path (str): 输出TXT文件的文件夹路径
        class_list (list, optional): 类别名称列表
    """
    
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    # 查找所有JSON文件
    json_files = glob.glob(os.path.join(json_folder_path, "*.json"))
    
    if not json_files:
        print(f"在文件夹 {json_folder_path} 中未找到JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件，开始转换...")
    
    # 批量转换
    success_count = 0
    for json_file in json_files:
        # 生成对应的TXT文件名
        base_name = os.path.basename(json_file).replace('.json', '.txt')
        output_txt_file = os.path.join(output_folder_path, base_name)
        
        # 转换单个文件
        try:
            convert_seg_json_to_det_txt(json_file, output_txt_file, class_list)
            success_count += 1
        except Exception as e:
            print(f"转换失败: {json_file}, 错误: {e}")
            continue
    
    print(f"批量转换完成！成功转换 {success_count}/{len(json_files)} 个文件")
    
    # 如果使用了类别列表，保存类别文件
    if class_list is not None:
        classes_file = os.path.join(output_folder_path, 'classes.txt')
        with open(classes_file, 'w', encoding='utf-8') as f:
            for i, class_name in enumerate(class_list):
                f.write(f"{class_name}\n")
        print(f"类别文件已保存: {classes_file}")

def validate_conversion(image_folder, label_folder, sample_count=3):
    """
    验证转换结果，显示几个样本的检测框 [2](@ref)
    
    Args:
        image_folder (str): 图像文件夹路径
        label_folder (str): 标签文件夹路径
        sample_count (int): 要验证的样本数量
    """
    import cv2
    import numpy as np
    
    # 获取图像文件列表
    image_files = [f for f in os.listdir(image_folder) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print("未找到图像文件进行验证")
        return
    
    print(f"\n验证转换结果（显示前{sample_count}个样本）:")
    
    for i, image_file in enumerate(image_files[:sample_count]):
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(label_folder, 
                                 os.path.splitext(image_file)[0] + '.txt')
        
        if not os.path.exists(label_path):
            print(f"未找到对应的标签文件: {label_path}")
            continue
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            continue
        
        h, w = image.shape[:2]
        
        # 读取标签
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # 在图像上绘制检测框
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            class_id, x_center, y_center, width, height = map(float, parts)
            
            # 转换回像素坐标
            x_center_px = int(x_center * w)
            y_center_px = int(y_center * h)
            width_px = int(width * w)
            height_px = int(height * h)
            
            x1 = int(x_center_px - width_px / 2)
            y1 = int(y_center_px - height_px / 2)
            x2 = int(x_center_px + width_px / 2)
            y2 = int(y_center_px + height_px / 2)
            
            # 绘制矩形框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'Class {int(class_id)}', (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 显示图像
        cv2.imshow(f'Validation: {image_file}', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# 使用示例
if __name__ == "__main__":
    # 设置类别列表（根据您的实际类别修改）
    class_names = ["0"]  # 替换为您的实际类别名称
    
    # 设置路径
    json_folder = r"E:\data\已标注\正面的四个侧面_缺陷检测_1217"  # JSON文件夹路径
    txt_folder = r"E:\data\已标注\正面的四个侧面_1217_det"  # 输出TXT文件夹路径
    
    # 批量转换
    batch_convert_seg_json_to_det_txt(json_folder, txt_folder, class_names)
    
    # 可选：验证转换结果（需要图像文件夹）
    # image_folder = r"E:\data\已标注\正面的四个侧面_缺陷检测_1217\images"  # 图像文件夹路径
    # validate_conversion(image_folder, txt_folder, sample_count=3)