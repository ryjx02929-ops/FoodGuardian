from ultralytics import YOLO

# 1. 載入輕量級的 YOLOv8s-World 模型
model = YOLO('yolov8s-world.pt')

# 2. 定義你的實體模板清單 (與網頁版相同，但獨立寫在這裡)
custom_food_list = [
    "Cabbage", "Chinese Cabbage", "Bok Choy", "Spinach", "Water Spinach", "Sweet Potato Leaves", "A-Choy", "Basil",
    "Green Onion", "Whole Garlic Bulb", "Ginger", "Chili", "Onion", "Green Pepper", "Bell Pepper", "Cucumber", "Sponge Gourd",
    "Bitter Gourd", "Winter Melon", "Pumpkin", "Zucchini", "Eggplant", "Tomato", "Okra", "Corn", "Baby Corn",
    "Carrot", "White Radish", "Potato", "Sweet Potato", "Taro", "Bamboo Shoot", "Water Bamboo", "Asparagus",
    "Bean Sprout", "Green Bean", "Edamame", "Shiitake", "Enoki Mushroom", "King Oyster Mushroom", "Broccoli",
    "Red Apple", "Banana", "Pineapple", "Papaya", "Watermelon", "Melon", "Grape", "Strawberry", "Orange", "Tangerine",
    "Lemon", "Passion Fruit", "Mango", "Guava", "Pear", "Dragon Fruit", "Kiwi", "Peach", "Cherry",
    "Pork", "Pork Belly", "Pork Chop", "Chicken", "Chicken Breast", "Chicken Leg", "Beef", "Steak", "Fish", "Salmon", "Shrimp",
    "Squid", "Clam", "Crab", "Egg", "Tofu", "Dried Tofu", "Soy Milk", "Milk", "Cheese", "Butter", "Flour", "Sugar", "Salt",
    "Soy Sauce", "Vinegar", "Cooking Oil", "Sesame Oil", "Olive Oil", "Rice", "Noodles", "Bread",
    "Wood Ear Mushroom", "Dried Shrimp", "Scallop", "Fermented Black Beans", "Doubanjiang"
]

# 3. 將清單綁定到模型中
model.set_classes(custom_food_list)

# 4. 匯出為 Xcode 專用的 CoreML 格式
# (參數 nms=True 極度重要！它會把過濾方框的演算法直接寫死在模型內，後續 iOS 開發會省事非常多)
print("正在轉換為 Apple CoreML 格式，請稍候...")
model.export(format='coreml', nms=True)
print("轉換成功！你可以準備開啟 Xcode 了。")
