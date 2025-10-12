import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Không mở được camera.")
else:
    print("✅ Camera đã mở.")
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Test Camera", frame)
        cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
