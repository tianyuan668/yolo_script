import json
import os
import glob

def convert_json_to_yolo_txt(json_file_path, output_txt_path, class_list=None):
    """
    将JSON分割标注文件转换为YOLO格式的TXT文件
    
    Args:
        json_file_path (str): 输入的JSON文件路径
        output_txt_path (str): 输出的TXT文件路径
        class_list (list, optional): 类别名称列表。如果为None，则使用JSON中的label作为类别ID
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
                
                # 归一化坐标点
                normalized_points = []
                for point in points:
                    x, y = point
                    # 归一化到 [0, 1] 范围
                    x_norm = x / image_width
                    y_norm = y / image_height
                    
                    # 确保坐标在有效范围内
                    x_norm = max(0.0, min(1.0, x_norm))
                    y_norm = max(0.0, min(1.0, y_norm))
                    
                    normalized_points.extend([x_norm, y_norm])
                
                # 写入TXT文件：类别ID + 归一化后的点坐标
                points_str = ' '.join([f'{coord:.6f}' for coord in normalized_points])
                txt_file.write(f"{class_id} {points_str}\n")
        
        print(f"成功转换: {json_file_path} -> {output_txt_path}")
        
    except Exception as e:
        print(f"转换失败 {json_file_path}: {str(e)}")

def batch_convert_json_to_txt(json_folder_path, output_folder_path, class_list=None):
    """
    批量转换文件夹中的所有JSON文件为YOLO TXT格式
    
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
    for json_file in json_files:
        # 生成对应的TXT文件名
        base_name = os.path.basename(json_file).replace('.json', '.txt')
        output_txt_file = os.path.join(output_folder_path, base_name)
        
        # 转换单个文件
        convert_json_to_yolo_txt(json_file, output_txt_file, class_list)
    
    print("批量转换完成！")
    
    # 如果使用了类别列表，保存类别文件
    if class_list is not None:
        classes_file = os.path.join(output_folder_path, 'classes.txt')
        with open(classes_file, 'w', encoding='utf-8') as f:
            for i, class_name in enumerate(class_list):
                f.write(f"{class_name}\n")
        print(f"类别文件已保存: {classes_file}")

# 使用示例
if __name__ == "__main__":
    # 示例1: 转换单个文件
    # 您的JSON数据中标签为"0"，这里假设类别0对应实际名称
    class_names = ["0"]  # 根据您的实际类别修改
    
    
    
    # 示例2: 批量转换整个文件夹
    json_folder = r"E:\data\已标注\正面的四个侧面_缺陷检测_1217"       # 替换为您的JSON文件夹路径
    txt_folder = r"E:\data\已标注\正面的四个侧面_1217"      # 替换为您的输出文件夹路径
    
    # 批量转换
    batch_convert_json_to_txt(json_folder, txt_folder, class_names)