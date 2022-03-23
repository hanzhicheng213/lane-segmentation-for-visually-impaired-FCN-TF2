import cv2
import tensorflow as tf
import numpy as np

scal = 224
video_path = r".\mangdao.mp4"

def normal_img(input_images):
    input_images = tf.cast(input_images, tf.float32)
    input_images = input_images/127.5 - 1
    input_images = tf.expand_dims(input_images, axis=0)
    return input_images

model = tf.keras.models.load_model('FCN.h5')

cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps =cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('out.avi', fourcc, fps, (224, 224), isColor=False)
# cap.isOpened()返回视频是否读取成功
while cap.isOpened():
    ret, frame = cap.read()  # ret返回视频是否获取成功， frame为每一帧图像
    if ret == True:
        frame = cv2.resize(frame, (224, 224))
        frame = frame.astype(np.uint8)
        frame = normal_img(frame)
        pred_frame = model.predict(frame)
        pred_frame = tf.argmax(pred_frame, axis=-1)
        pred_frame = pred_frame * 255
        pred_frame = pred_frame[..., tf.newaxis]
        pred_frame = tf.squeeze(pred_frame)
        pred_frame = np.array(pred_frame)
        pred_frame = pred_frame.astype(np.uint8)
        out.write(pred_frame)
        cv2.imshow("pred_frame", pred_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()


