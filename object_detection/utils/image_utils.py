import cv2
def detect_image(model, img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Không đọc được ảnh: {img_path}")
        return None

    results = model.predict(img, verbose=False)
    annotated_img = results[0].plot()
    return annotated_img


def save_image(img, output_path):
    cv2.imwrite(output_path, img)
    print(f"Đã lưu tại: {output_path}")
