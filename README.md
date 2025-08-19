# 🍱 食物影像辨識系統

一個基於深度學習的 **食物影像分類模型**（共 11 類）。
本專案使用 **Python + PyTorch**，並以 **ResNet-18** 作為骨幹模型（backbone），透過殘差連接有效穩定梯度傳遞。
在預訓練階段於驗證集達到 **約 75% Top-1 準確率**，作為後續微調 (fine-tuning) 的基礎。

---

## 📌 功能特色

* 🍜 **分類 11 種食物影像**
* 🧠 **ResNet-18 架構**：利用殘差連接減緩梯度消失/爆炸
* 📈 **驗證集約 75% 準確率**（Top-1）
* 🔧 可進一步進行 **微調 (fine-tuning)** 應用於不同資料集

---

## 🛠️ 技術棧

* **程式語言**：Python
* **深度學習框架**：PyTorch
* **模型**：ResNet-18 (預訓練骨幹模型)
* **資料集**：食物影像資料集（11 類別）

---

## 📂 專案結構

```
food_image_identification/
│── Dataset/              # 資料集 (影像 / 標籤)
│── food_image_identification.py/               # 模型定義與訓練腳本
│── pratical_test.py/           # 訓練結果與模型權重
|──submission.csv               # 提交給Kaggle以便查看測驗結果
│── README.md                   # 專案說明文件
```

---

## 🚀 使用方式

### 1️⃣ 下載專案

```bash
git clone https://github.com/your-username/food_image_identification.git
cd food_image_identification
```

### 2️⃣ 安裝依賴套件

```bash
pip install -r requirements.txt
```

### 3️⃣ 訓練模型

```bash
python food_image_identification.py
```

### 4️⃣ 驗證模型

```bash
python pratical_test.py --model outputs/model.pth
```

---

## 📊 結果

* **模型**：ResNet-18
* **任務**：食物影像分類（11 類）
* **驗證集準確率**：Top-1 約 75%

---

## 📜 授權

本專案僅供 **學習與研究用途**。

---


