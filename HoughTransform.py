import torch
import pathlib
import cv2
import requests  # 用於發送 LINE Notify 訊息

# 修正 pathlib 的路徑問題
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# LINE Notify 的 Token（需要替換為你的 Token）
LINE_NOTIFY_TOKEN = '6F23SilsdeW3vlPVEGB3tlM5qWv8HW0hCwIMhPQ4AIn'

def send_line_notification(message):
    """
    發送 LINE Notify 訊息
    """
    url = 'https://notify-api.line.me/api/notify'
    headers = {
        'Authorization': f'Bearer {LINE_NOTIFY_TOKEN}'
    }
    data = {
        'message': message
    }
    try:
        response = requests.post(url, headers=headers, data=data)
        if response.status_code == 200:
            print("LINE 訊息發送成功")
        else:
            print(f"LINE 訊息發送失敗: {response.status_code}")
    except Exception as e:
        print(f"無法發送 LINE 訊息: {e}")

# 加载 YOLOv5 模型
try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5-master/runs/train/exp6/weights/best.pt')
except Exception as e:
    print(f"加载 YOLOv5 模型失败: {e}")
    exit()

# 打開攝像頭
cap = cv2.VideoCapture(0)  # '0' 表示默認攝像頭
if not cap.isOpened():
    print("無法打開攝像頭，請檢查設備")
    exit()

print("按下 'q' 鍵退出")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法從攝像頭讀取畫面")
            break

        # YOLOv5 模型推理
        try:
            results = model(frame)
            detections = results.pandas().xyxy[0]  # 取得檢測結果的 DataFrame
        except Exception as e:
            print(f"推理過程中出錯: {e}")
            break

        # 檢查是否有 "taco" 的檢測結果
        if not detections.empty:
            for _, row in detections.iterrows():
                if row['name'] == 'taco':  # 假設你的模型標籤中 'taco' 為正確名稱
                    print("檢測到 taco!")
                    send_line_notification("發現 taco！")
                    break

        # 繪製檢測結果
        annotated_frame = results.render()[0]  # results.render() 返回帶有標註的圖片

        # 顯示結果
        cv2.imshow('YOLOv5 Detection', annotated_frame)

        # 按下 'q' 鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("退出 YOLOv5 檢測")
            break
finally:
    # 確保釋放攝像頭並關閉窗口
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    print("資源已釋放，窗口已關閉")






