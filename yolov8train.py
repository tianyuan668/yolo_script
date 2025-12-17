from ultralytics import YOLO

if __name__ == "__main__":
    # 初始化模型（加载预训练权重）
    model = YOLO('yolov8n.pt')  # 可选：yolov8s.pt, yolov8m.pt等[1,5](@ref)

    # 开始训练
    results = model.train(
        data=r'E:\data\正面的四个侧面/dataset.yaml',  # 数据集配置文件路径[1,3](@ref)
        epochs=300,  # 训练轮次
        imgsz=1024,  # 输入图像尺寸
        batch=16,  # 批量大小（根据GPU显存调整）[5,8](@ref)
        save=True,  # 保存检查点
        save_period=100,  # 每10个epoch保存模型
        device='0',
        cache=False,  # 禁用数据缓存
        project='runs/train',  # 结果保存路径
        name='corn_leaf_detection'  # 实验名称
    )

    # 保存最终模型
    model.save('best_model.pt')
    print("Training completed and model saved!")