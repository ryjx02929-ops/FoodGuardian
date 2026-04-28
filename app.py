import traceback
import cv2
import numpy as np
import base64
from ultralytics import YOLO
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
import torch

# 👇 新增 LINE 與 Firebase 的套件
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ==========================================
# 1. LINE Bot 與 Firebase 初始化設定
# ==========================================
# ⚠️ 請換成你在 LINE Developers > Messaging API 拿到的金鑰
line_bot_api = LineBotApi('a9uZGfT+NPtcD0zi/7hQr7Oh1RnZgY4qPombJyhhGasT7v8Ki5isp1x4PfMk3ap5wVybz1B4QVsy4g4WwnYuHNRjM0q9cNAauYiE7vBl6iRpu+EB6DBxq7TwWWV6DHfXRkTN2YTkFI5eo905ChYjFwdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('7738c0704ea63f77dcd8d9ece7924773')

# 初始化 Firebase Admin (讀取剛剛下載的 JSON 鑰匙)
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# --- 1. 官方模型絕對不能改的 80 個順序 (確保對位準確) ---
custom_food_list = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# --- 2. 專題專用翻譯字典 (我們只翻譯食物，其他的當作沒看到) ---
translation_dict = {
    "apple": "蘋果",
    "orange": "橘子",  # 👈 你的橘子在這裡！
    "banana": "香蕉",
    "broccoli": "青花菜",
    "carrot": "紅蘿蔔",
    "sandwich": "三明治",
    "pizza": "披薩",
    "cake": "蛋糕",
    "donut": "甜甜圈",
    "hot dog": "熱狗",
    "bottle": "瓶裝飲料",
    "cup": "杯裝飲品",
    "bowl": "碗裝食物"
}

model = YOLO('yolov8n.onnx', task='detect')
torch.set_num_threads(1)

@app.route('/scan_image', methods=['POST'])
def scan_image():
    try:
        # 1. 確保有收到正確的 JSON 與圖片資料
        json_data = request.get_json()
        if not json_data or 'image' not in json_data:
            print("🚨 錯誤：缺少 image 欄位或非 JSON 格式")
            return jsonify({"error": "Missing image data"}), 400

        data = json_data['image']
        if ',' not in data:
            print("🚨 錯誤：圖片 Base64 格式不正確")
            return jsonify({"error": "Invalid format"}), 400

        # 2. 解析 base64 並轉換為 OpenCV 圖片
        encoded_data = data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 3. 進行 YOLO 辨識
        # 加入 imgsz=320 強制 AI 將運算解析度降到最低，避免 512MB 記憶體爆炸
        results = model(img, conf=0.15, iou=0.3, imgsz=320)
        inventory_count = {}

        # 4. 統計辨識結果 (過濾系統)
        for r in results:
            for box in r.boxes:
                cls_idx = int(box.cls[0])
                if cls_idx < len(custom_food_list):
                    eng_name = custom_food_list[cls_idx]
                    
                    # 只有在翻譯字典裡的「食物」才會被加入清單
                    if eng_name in translation_dict:
                        zh_name = translation_dict[eng_name]
                        inventory_count[zh_name] = inventory_count.get(zh_name, 0) + 1
                    
        return jsonify(inventory_count)

    except Exception as e:
        print("🚨 發生嚴重錯誤:", str(e))
        print(traceback.format_exc()) 
        return jsonify({"error": "伺服器內部錯誤", "details": str(e)}), 500

# ==========================================
# 4. 處理使用者在 LINE 傳送的文字訊息
# ==========================================
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_line_id = event.source.user_id
    user_text = event.message.text.strip()

    # 檢查使用者是否已經綁定過帳號
    bind_doc = db.collection('line_bindings').document(user_line_id).get()
    
    if not bind_doc.exists:
        # 如果沒綁定，給他你的 LIFF 綁定網址！
        # ⚠️ 記得把下面的 LIFF_ID 換成你的
        liff_url = "https://liff.line.me/你的_LIFF_ID"
        reply_msg = f"哈囉！你還沒有綁定食光守護者的冰箱喔！\n請點擊下方連結進行綁定：\n{liff_url}"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_msg))
        return

    firebase_uid = bind_doc.to_dict().get('firebaseUid')
    user_doc = db.collection('users').document(firebase_uid).get()
    
    if not user_doc.exists:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="你的冰箱目前空空如也，快去網頁版新增食材吧！"))
        return
        
    inventory = user_doc.to_dict().get('inventory', [])

    # --- 關鍵字回覆邏輯 ---
    if user_text in ["冰箱狀態", "查冰箱", "快過期的食材"]:
        today = datetime.now().date()
        expired_items = []
        warning_items = []

        for item in inventory:
            # 將字串 "2026-04-25" 轉成日期物件
            expiry_date = datetime.strptime(item.get('expiry'), "%Y-%m-%d").date()
            diff_days = (expiry_date - today).days

            if diff_days < 0:
                expired_items.append(f"❌ {item.get('name')} (已過期)")
            elif diff_days <= 3:
                warning_items.append(f"⚠️ {item.get('name')} (剩 {diff_days} 天)")

        # 組合回覆訊息
        reply_lines = []
        if not expired_items and not warning_items:
            reply_lines.append("✨ 目前冰箱裡的食材都很新鮮喔！")
        else:
            if expired_items:
                reply_lines.append("【已過期請丟棄】")
                reply_lines.extend(expired_items)
                reply_lines.append("")
            if warning_items:
                reply_lines.append("【即期品請優先食用】")
                reply_lines.extend(warning_items)

        reply_text = "\n".join(reply_lines)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

    elif user_text == "購物清單":
        shopping_data = user_doc.to_dict().get('shoppingData', [])
        unbought = [item.get('name') for item in shopping_data if not item.get('checked')]
        
        if unbought:
            reply_text = "🛒 你的待買清單有：\n" + "\n".join([f"• {name}" for name in unbought])
        else:
            reply_text = "🛒 目前沒有待買物品喔！"
            
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

    else:
        # 使用者輸入其他文字時的預設回覆
        reply_text = "你可以對我說：\n🔍 「冰箱狀態」\n🛒 「購物清單」"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
