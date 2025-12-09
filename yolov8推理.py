from ultralytics import YOLO
import cv2

# 加载预训练模型（替换为你的模型路径）
model = YOLO("yolo11x.pt")  # 支持.pt或.onnx格式


# 图片检测（单张）
def detect_image(img_path):
    results = model.predict(img_path, conf=0.5,imgsz=1024, save=False)  # 关键参数：置信度阈值
    for result in results:
        result.show()  # 显示带检测框的图像
        # result.save("output.jpg")  # 保存结果


# 视频检测（实时摄像头或视频文件）
def detect_video(video_path=0):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # 推理并绘制结果
        results = model.predict(frame, imgsz=1024, verbose=False)
        annotated_frame = results[0].plot()

        # 显示结果
        cv2.imshow("YOLOv11 Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


# 使用示例
if __name__ == "__main__":
    detect_image("1.jpg")  # 取消注释运行图片检测
    #detect_video()  # 默认使用摄像头，可传入视频路径