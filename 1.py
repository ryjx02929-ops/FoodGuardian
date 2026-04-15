import cv2
from ultralytics import YOLO
import cvzone
import math

# 1. 載入模型
# 使用 'yolov8n.pt' (Nano版本)，速度最快，適合即時偵測
# 如果你的電腦有顯卡且想要更準確，可以改用 'yolov8m.pt'
model = YOLO('yolov8n.pt')

# 2. 設定辨識類別名稱 (COCO 資料集標準 80 類)
# 我們主要關注食材類，例如：apple, orange, broccoli, carrot, hot dog, pizza, donut, cake
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# 你可以在這裡定義只想要統計的食材清單
target_foods = ["apple", "banana", "orange", "broccoli", "carrot", "cake", "donut", "pizza"]

# 3. 開啟攝影機
# 若是要串接 ESP32-CAM，請將 0 改為網址字串，例如 'http://192.168.1.105:81/stream'
cap = cv2.VideoCapture(0)
cap.set(3, 1280) # 設定寬度
cap.set(4, 720)  # 設定高度

while True:
    success, img = cap.read()
    if not success:
        break

    # 4. 進行偵測 (stream=True 讓串流更順暢)
    results = model(img, stream=True)

    # 用來統計當前畫面的食材數量
    inventory_count = {} 

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # 取得類別 ID
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # 篩選：只偵測我們感興趣的食材，且信心度 > 0.5
            conf = math.ceil((box.conf[0] * 100)) / 100
            if currentClass in target_foods and conf > 0.5:
                
                # --- 統計數量 ---
                if currentClass in inventory_count:
                    inventory_count[currentClass] += 1
                else:
                    inventory_count[currentClass] = 1

                # --- 畫圖 ---
                # 取得座標 (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # 畫框框 (顏色為綠色，寬度 3)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # 顯示標籤文字 (例如 Apple 0.85)
                cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), 
                                   scale=1.5, thickness=2, colorR=(0, 200, 0), offset=5)

    # 5. 製作介面：顯示「Inventory List」(模擬圖片左上角的黑色區塊)
    # 畫一個半透明黑色背景區塊在左上角
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (250, 400), (0, 0, 0), -1) # 黑色實心矩形
    alpha = 0.6 # 透明度
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # 寫上標題
    cv2.putText(img, "Inventory List:", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 逐行列出統計到的食材
    y_pos = 80
    if not inventory_count:
        cv2.putText(img, "No food detected", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    else:
        for food, count in inventory_count.items():
            # 格式： apple: 4 (黃色文字)
            text = f"{food}: {count}"
            cv2.putText(img, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            y_pos += 40

    # 6. 顯示結果視窗
    cv2.imshow("Smart_Food_Checkout", img)

    # 按下 'q' 鍵離開
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()