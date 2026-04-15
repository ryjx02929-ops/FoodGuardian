import cv2
import numpy as np
import base64
from ultralytics import YOLO
import math
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- 1. 擴充版食材清單 (加入精準肉類描述) ---
# 秘訣：使用 "Raw" (生的), "Sliced" (切片的), "Packaged" (包裝好的) 來引導 AI 尋找超市肉類，而非活體動物。
custom_food_list = [
    # 蔬菜與水果 (保留原有)
    "Cabbage", "Chinese Cabbage", "Bok Choy", "Spinach", "Water Spinach", "Sweet Potato Leaves", "A-Choy", "Basil", 
    "Green Onion", "Whole Garlic Bulb", "Ginger", "Chili", "Onion", "Green Pepper", "Bell Pepper", "Cucumber", "Sponge Gourd", 
    "Bitter Gourd", "Winter Melon", "Pumpkin", "Zucchini", "Eggplant", "Tomato", "Okra", "Corn", "Baby Corn", 
    "Carrot", "White Radish", "Potato", "Sweet Potato", "Taro", "Bamboo Shoot", "Water Bamboo", "Asparagus", 
    "Bean Sprout", "Green Bean", "Edamame", "Shiitake", "Enoki Mushroom", "King Oyster Mushroom", "Broccoli",
    "Red Apple", "Banana", "Pineapple", "Papaya", "Watermelon", "Melon", "Grape", "Strawberry", "Orange", "Tangerine", 
    "Lemon", "Passion Fruit", "Mango", "Guava", "Pear", "Dragon Fruit", "Kiwi", "Peach", "Cherry",
    
    # 🔥 優化後的肉類清單 (多重描述增加命中率)
    "Raw pork meat", "Packaged raw pork", "Raw pork belly slice", "Raw pork chop", 
    "Sausage", "Bacon", "Ham", 
    "Raw beef meat", "Raw beef steak", "Sliced raw beef", "Packaged raw beef","Meat", "Beef slice", "Red meat",
    "Raw chicken meat", "Raw chicken breast", "Raw chicken leg", "Raw chicken wing", 
    "Raw salmon fillet", "Raw tuna fillet", "Raw cod fish", "Raw mackerel", "Raw tilapia fish", 
    "Raw shrimp", "Raw clam", "Raw oyster",
    
    # 主食、烘焙與調味 (保留原有)
    "Rice", "Noodle", "Pasta", "Toast", "Bread", "Bun", "Dumpling", "Egg", "Tofu", "Dried Tofu", 
    "Bean Curd Skin", "Milk", "Soy Milk", "Oat Milk", "Yogurt", "Cheese", "Butter", "Soy Sauce", "Ketchup", "Kimchi",
    "Flour", "Sugar", "Yeast", "Baking Powder", "Vanilla Extract", "Chocolate", "Whipping Cream", "Cream Cheese", "Almond Flour", "Matcha Powder",
    "Sesame Oil", "Oyster Sauce", "Rice Wine", "Star Anise", "Wood Ear Mushroom", "Dried Shrimp", "Scallop", "Fermented Black Beans", "Doubanjiang",
    "Tuna can", "Kimchi can", "Spam", "Ham slice", "Bacon slice",
    "Coke", "Pepsi", "Milk tea", "Oolong tea", "Green tea",
    "Udon noodle", "Ramen", "Instant noodle", "Pasta",
    "Strawberry", "Blueberry", "Cherry", "Grapes",
    "Sausage", "Hot dog", "Chicken nugget", "Fish ball"
]

# --- 2. 食材英翻中字典 (多對一映射) ---
translation_dict = {
    # 蔬菜與水果
    "Cabbage": "高麗菜", "Chinese Cabbage": "大白菜", "Bok Choy": "小白菜", "Spinach": "菠菜", "Water Spinach": "空心菜", 
    "Sweet Potato Leaves": "地瓜葉", "A-Choy": "A菜", "Basil": "九層塔", "Green Onion": "青蔥", "Whole Garlic Bulb": "蒜頭", 
    "Ginger": "薑", "Chili": "辣椒", "Onion": "洋蔥", "Green Pepper": "青椒", "Bell Pepper": "甜椒", "Cucumber": "小黃瓜", 
    "Sponge Gourd": "絲瓜", "Bitter Gourd": "苦瓜", "Winter Melon": "冬瓜", "Pumpkin": "南瓜", "Zucchini": "櫛瓜", 
    "Eggplant": "茄子", "Tomato": "番茄", "Okra": "秋葵", "Corn": "玉米", "Baby Corn": "玉米筍", "Carrot": "紅蘿蔔", 
    "White Radish": "白蘿蔔", "Potato": "馬鈴薯", "Sweet Potato": "地瓜", "Taro": "芋頭", "Bamboo Shoot": "竹筍", 
    "Water Bamboo": "筊白筍", "Asparagus": "蘆筍", "Bean Sprout": "豆芽菜", "Green Bean": "四季豆", "Edamame": "毛豆", 
    "Shiitake": "香菇", "Enoki Mushroom": "金針菇", "King Oyster Mushroom": "杏鮑菇", "Broccoli": "花椰菜",
    "Red Apple": "蘋果", "Banana": "香蕉", "Pineapple": "鳳梨", "Papaya": "木瓜", "Watermelon": "西瓜", "Melon": "哈密瓜", 
    "Grape": "葡萄", "Strawberry": "草莓", "Orange": "柳丁", "Tangerine": "橘子", "Lemon": "檸檬", "Passion Fruit": "百香果", 
    "Mango": "芒果", "Guava": "芭樂", "Pear": "水梨", "Dragon Fruit": "火龍果", "Kiwi": "奇異果", "Peach": "桃子", "Cherry": "櫻桃", 
    
    # 🔥 優化後的肉類對應
    "Raw pork meat": "豬肉", "Packaged raw pork": "豬肉",
    "Raw pork belly slice": "豬五花", 
    "Raw pork chop": "豬排", 
    "Sausage": "香腸", "Bacon": "培根", "Ham": "火腿", 
    "Raw beef meat": "牛肉", "Sliced raw beef": "牛肉", "Packaged raw beef": "牛肉","Meat": "牛肉", "Beef slice": "牛肉", "Red meat": "牛肉", # 👈 新增對應
    "Raw beef steak": "牛排", 
    "Wagyu beef": "牛肉", "Marbled beef": "牛肉", "Raw red meat": "牛肉", # 👈 新增這行對應
    
    # 主食、烘焙與調味
    "Rice": "白米", "Noodle": "麵條", "Pasta": "義大利麵", "Toast": "吐司", "Bread": "麵包", "Bun": "包子", 
    "Dumpling": "水餃", "Egg": "雞蛋", "Tofu": "豆腐", "Dried Tofu": "豆乾", "Bean Curd Skin": "豆皮", 
    "Milk": "牛奶", "Soy Milk": "豆漿", "Oat Milk": "燕麥奶", "Yogurt": "優格", "Cheese": "起司", "Butter": "奶油", 
    "Soy Sauce": "醬油", "Ketchup": "番茄醬", "Kimchi": "泡菜",
    "Flour": "麵粉", "Sugar": "砂糖", "Yeast": "酵母粉", "Baking Powder": "泡打粉", "Vanilla Extract": "香草精", 
    "Chocolate": "巧克力", "Whipping Cream": "鮮奶油", "Cream Cheese": "奶油乳酪", "Almond Flour": "杏仁粉", "Matcha Powder": "抹茶粉",
    "Sesame Oil": "香油", "Oyster Sauce": "蠔油", "Rice Wine": "米酒", "Star Anise": "八角", 
    "Wood Ear Mushroom": "木耳", "Dried Shrimp": "蝦米", "Scallop": "干貝", "Fermented Black Beans": "豆豉", "Doubanjiang": "豆瓣醬",
    "Tuna can": "鮪魚罐頭", "Kimchi can": "泡菜罐頭", "Spam": "午餐肉",
    "Coke": "可樂", "Milk tea": "奶茶", "Udon noodle": "烏龍麵",
    "Chicken nugget": "雞塊", "Fish ball": "魚丸"
}

# 初始化模型
model = YOLO('yolov8l-world.pt')
model.set_classes(custom_food_list)

@app.route('/scan_image', methods=['POST'])
def scan_image():
    try:
        # 1. 接收前端傳來的 base64 圖片資料
        data = request.json.get('image')
        if not data:
            return jsonify({"error": "No image data"}), 400

        # 2. 解析 base64 並轉換為 OpenCV 可讀取的格式
        encoded_data = data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 3. 進行 YOLO 辨識
        # 🔥 核彈級降維打擊：門檻降到 0.15
        results = model(img, conf=0.15, iou=0.3)
        inventory_count = {}

        # 4. 統計辨識結果
        for r in results:
            for box in r.boxes:
                cls_idx = int(box.cls[0])
                if cls_idx < len(custom_food_list):
                    eng_name = custom_food_list[cls_idx]
                    # 透過字典將複雜的英文描述對應回標準中文
                    zh_name = translation_dict.get(eng_name, eng_name)
                    inventory_count[zh_name] = inventory_count.get(zh_name, 0) + 1
                    
        # 5. 回傳 JSON 清單給前端
        return jsonify(inventory_count)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # 啟動伺服器
    app.run(host='0.0.0.0', port=5000, debug=False)