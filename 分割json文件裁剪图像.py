import json
import os
import cv2
import numpy as np
from pathlib import Path

class NumpyEncoder(json.JSONEncoder):
    """
    自定义 JSON 编码器，用于处理 NumPy 数据类型
    解决 'Object of type int64 is not JSON serializable' 错误[1,2,3](@ref)
    """
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64,
                          np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.datetime64, np.complex_)):
            return str(obj)
        else:
            return super(NumpyEncoder, self).default(obj)

def cv_imread(file_path):
    """安全读取中文路径图片"""
    try:
        with open(file_path, "rb") as f:
            return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"读取失败: {str(e)}")
        return None

def cv_imwrite(save_path, img):
    """保存图片到中文路径"""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ext = os.path.splitext(save_path)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            ext = '.jpg'
        
        ret, buf = cv2.imencode(ext, img)
        if ret:
            with open(save_path, "wb") as f:
                f.write(buf.tobytes())
            return True
        return False
    except Exception as e:
        print(f"保存失败 {save_path}: {str(e)}")
        return False

def calculate_min_area_rect(points):
    """根据多边形点计算最小外接矩形"""
    points = np.array(points, dtype=np.float32)
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box = np.int64(box)
    
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    
    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)
    
    w = x2 - x1
    h = y2 - y1
    
    return x1, y1, w, h

def update_json_coordinates(original_json_data, crop_x, crop_y, crop_width, crop_height, object_index=None):
    """
    更新JSON中的坐标点，使其适应裁剪后的图像
    使用自定义编码器解决 NumPy 类型序列化问题[1,4](@ref)
    """
    updated_json_data = original_json_data.copy()
    updated_shapes = []
    
    # 处理每个形状
    for idx, shape in enumerate(original_json_data.get('shapes', [])):
        if object_index is not None and idx != object_index:
            continue
            
        points = shape.get('points', [])
        updated_points = []
        
        # 更新每个点的坐标
        for point in points:
            # 转换为 Python 原生类型，避免序列化问题[3](@ref)
            new_x = float(point[0] - crop_x)  # 显式转换为 float
            new_y = float(point[1] - crop_y)
            
            new_x = max(0.0, min(float(crop_width), new_x))
            new_y = max(0.0, min(float(crop_height), new_y))
            
            updated_points.append([new_x, new_y])
        
        updated_shape = shape.copy()
        updated_shape['points'] = updated_points
        updated_shapes.append(updated_shape)
    
    updated_json_data['shapes'] = updated_shapes
    # 确保尺寸为 Python 原生类型
    updated_json_data['imageHeight'] = int(crop_height)
    updated_json_data['imageWidth'] = int(crop_width)
    updated_json_data['imageData'] = None
    
    return updated_json_data

def crop_image_by_json(json_path, image_path, output_folder):
    """
    根据JSON文件中的分割点裁剪图像，并生成对应的更新后的JSON文件
    修复了 NumPy 类型序列化问题[1,2](@ref)
    """
    try:
        # 读取JSON文件
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 使用支持中文路径的函数读取图像
        image = cv_imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return 0
        
        # 获取图像尺寸并确保为 Python 原生类型
        image_height = int(data.get('imageHeight', image.shape[0]))
        image_width = int(data.get('imageWidth', image.shape[1]))
        
        # 获取基础文件名（不含扩展名）
        base_name = Path(image_path).stem
        
        object_count = 0
        
        # 处理每个分割形状
        for i, shape in enumerate(data.get('shapes', [])):
            label = shape.get('label', 'object')
            points = shape.get('points', [])
            
            if len(points) < 3:
                print(f"跳过点数不足的形状: {label}")
                continue
            
            # 计算最小外接矩形
            x, y, w, h = calculate_min_area_rect(points)
            
            # 确保坐标为 Python 原生类型[3](@ref)
            x = int(max(0, x))
            y = int(max(0, y))
            w = int(min(w, image_width - x))
            h = int(min(h, image_height - y))
            
            if w <= 0 or h <= 0:
                print(f"无效的矩形尺寸: {w}x{h}，跳过")
                continue
            
            # 裁剪图像
            cropped_image = image[y:y+h, x:x+w]
            
            # 生成输出文件名
            output_filename = f"{base_name}_{label}_{i+1}_{w}x{h}"
            image_output_path = os.path.join(output_folder, f"{output_filename}.jpg")
            json_output_path = os.path.join(output_folder, f"{output_filename}.json")
            
            # 使用支持中文路径的函数保存图像
            if cv_imwrite(image_output_path, cropped_image):
                print(f"保存图像: {output_filename}.jpg")
                
                # 更新JSON数据
                updated_json_data = update_json_coordinates(data, x, y, w, h, i)
                updated_json_data['imagePath'] = f"{output_filename}.jpg"
                
                # 保存更新后的JSON文件，使用自定义编码器[1,4](@ref)
                try:
                    with open(json_output_path, 'w', encoding='utf-8') as f:
                        # 使用自定义的 NumpyEncoder 解决序列化问题
                        json.dump(updated_json_data, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
                    
                    print(f"保存JSON: {output_filename}.json")
                    object_count += 1
                except Exception as json_error:
                    print(f"保存JSON失败 {output_filename}.json: {str(json_error)}")
                    # 尝试备用方法：使用 default=str
                    try:
                        with open(json_output_path, 'w', encoding='utf-8') as f:
                            json.dump(updated_json_data, f, indent=4, ensure_ascii=False, default=str)
                        print(f"使用备用方法保存JSON: {output_filename}.json")
                        object_count += 1
                    except Exception as fallback_error:
                        print(f"备用方法也失败: {str(fallback_error)}")
            else:
                print(f"保存失败: {output_filename}.jpg")
            
        return object_count
            
    except Exception as e:
        print(f"处理文件 {json_path} 时出错: {str(e)}")
        return 0

# 其余函数保持不变（find_matching_pairs, process_with_pair_matching 等）
def find_matching_pairs(input_folder):
    """查找图像和JSON文件的匹配对"""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    
    image_files = {}
    json_files = {}
    
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        base_name = Path(filename).stem
        extension = Path(filename).suffix.lower()
        
        if extension in image_extensions:
            image_files[base_name] = file_path
        elif extension == '.json':
            json_files[base_name] = file_path
    
    pairs = []
    for base_name in image_files:
        if base_name in json_files:
            pairs.append((image_files[base_name], json_files[base_name]))
    
    return pairs

def process_with_pair_matching(input_folder, output_folder):
    """使用文件匹配的方式处理中文路径图像和JSON文件"""
    os.makedirs(output_folder, exist_ok=True)
    
    pairs = find_matching_pairs(input_folder)
    
    print(f"找到 {len(pairs)} 个匹配的图像-JSON文件对")
    
    processed_count = 0
    error_count = 0
    total_objects = 0
    
    for image_path, json_path in pairs:
        try:
            object_count = crop_image_by_json(json_path, image_path, output_folder)
            if object_count > 0:
                processed_count += 1
                total_objects += object_count
                print(f"成功处理: {Path(image_path).name}，提取了 {object_count} 个对象")
            else:
                print(f"{Path(image_path).name} 未提取到任何对象")
                error_count += 1
                
        except Exception as e:
            print(f"处理 {Path(image_path).name} 时出错: {str(e)}")
            error_count += 1
    
    print(f"\n处理完成！")
    print(f"成功处理: {processed_count} 个文件对")
    print(f"处理失败: {error_count} 个文件对")
    print(f"总共提取: {total_objects} 个对象")
    print(f"输出文件夹: {output_folder}")

# 使用示例
if __name__ == "__main__":
    # 设置路径（请根据实际情况修改）
    input_folder = r"Y:\01.软件数据组\SRTGZ-2025-ZB-008-深圳懿晗-美利信逆变器检测项目\现场数据\分图\检测\正常样本"
    output_folder = r"Y:\01.软件数据组\SRTGZ-2025-ZB-008-深圳懿晗-美利信逆变器检测项目\现场数据\分图\检测\裁剪结果"
    
    # 开始批量处理
    process_with_pair_matching(input_folder, output_folder)