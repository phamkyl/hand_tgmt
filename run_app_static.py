import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

# Đường dẫn đến mô hình đã lưu
model_path = 'tgmt_hand_v03_1.h5'
# Tải mô hình đã huấn luyện
model = load_model(model_path)

# Định nghĩa từ điển nhãn
label_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
    17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
    25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'
}


def predict_image(image_path):
    # Tải ảnh và xử lý ảnh
    new_img = image.load_img(image_path, target_size=(64, 64))
    img = image.img_to_array(new_img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Chuẩn hóa ảnh

    # Dự đoán
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)

    # In ra kết quả dự đoán
    predicted_label = label_dict[predicted_class[0]]
    print(f'Predicted label: {predicted_label}')

    return new_img, predicted_label


def select_image():
    # Mở hộp thoại chọn file
    file_path = filedialog.askopenfilename()
    if file_path:
        img, label = predict_image(file_path)
        display_image(img, label)


def display_image(img, label):
    # Tạo cửa sổ mới để hiển thị ảnh và kết quả dự đoán
    top = tk.Toplevel()
    top.title(f'Prediction: {label}')

    # Chuyển đổi ảnh từ định dạng PIL sang định dạng phù hợp với Tkinter
    img = Image.fromarray((img * 255).astype(np.uint8))
    img = img.resize((200, 200), Image.BILINEAR)
    imgtk = ImageTk.PhotoImage(image=img)

    label_img = tk.Label(top, image=imgtk)
    label_img.image = imgtk
    label_img.pack()

    label_text = tk.Label(top, text=f'Prediction: {label}', font=('Arial', 14))
    label_text.pack()


# Tạo cửa sổ chính
root = tk.Tk()
root.title("Sign Language Prediction")

# Tạo nút để mở hộp thoại chọn ảnh
btn = tk.Button(root, text="Select Image", command=select_image)
btn.pack()

# Chạy ứng dụng
root.mainloop()
