# 🩺 Breast Cancer Detection using Histopathological Images

A deep learning project that classifies breast histopathology images as **Benign** or **Malignant** using **PyTorch**, **ResNet18**, and a **Streamlit web application**.

---

## 🚀 Live Features

* Upload histopathology image
* Predict **Benign / Malignant**
* Confidence score display
* Clean Streamlit frontend
* GPU-supported model training
* Transfer learning using ResNet18

---

## 🧠 Model Used

* **ResNet18**
* Transfer Learning
* Final layer modified for binary classification

Classes:

* `0 → Benign`
* `1 → Malignant`

---

## 📂 Project Structure

```text
cancer-detection/
│── app.py
│── requirements.txt
│── README.md
│── models/
│   └── best_model.pth
│── src/
│── notebooks/
│── dataset/
```

---

## 🗃️ Dataset

Dataset used: **BreaKHis_v1**

Breast Cancer Histopathological Image Dataset containing:

* Benign tumor images
* Malignant tumor images

---

## ⚙️ Technologies Used

* Python
* PyTorch
* Torchvision
* Streamlit
* NumPy
* Matplotlib
* Scikit-learn
* VS Code

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

If app file is inside `src`:

```bash
streamlit run src/app.py
```

---

## 📊 Outputs

* Prediction label
* Confidence score
* Trained model weights
* Visualizations (loss / accuracy graphs)

---

## 🎯 Future Improvements

* Grad-CAM heatmap visualization
* Multi-class subtype detection
* Better UI dashboard
* Cloud deployment
* Doctor report generation

---

## 👨‍💻 Author

**Yash Mohan**

GitHub: https://github.com/YASHMOHAN1

---

## ⭐ If you like this project

Star this repository and share it.
