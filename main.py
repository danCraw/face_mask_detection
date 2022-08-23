from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils.image_utils import img_to_array
from keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
import os
from constants import MINIMUM_CERTAINTY, RED, GREEN


def build_face(frame, startY, endY, startX, endX):
    # извлекаием ROI лица, преобразуем его из BGR в RGB
    # упорядочивая, изменяем размер до 224x224 и выполняем предварительную обработку
    face = frame[startY:endY, startX:endX]
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    return face


def build_label(mask, withoutMask):
    label = "Mask" if mask > withoutMask else "No Mask"
    return "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)


def stop_detection(key) -> bool:
    # останавлваемся, если была нажата кнопка 'q'
    return key == ord("q")


def detect_mask(frame, faceNet, maskNet):
    # берем изображение из фрейма и получаем из него данные
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    # передаем массив данных и получаем распознанные лица
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    # инициализируем списки
    faces = []
    locs = []
    preds = []

    # цикл по обнаружению
    for i in range(0, detections.shape[2]):
        # получаем достоверность обнаружения
        confidence = detections[0, 0, i, 2]

        # фильтруем наши обнаружения, обрабатывая только те, достоверность которых больше минимальной
        if confidence > MINIMUM_CERTAINTY:
            # вычисляем координаты рамки для лица
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # устанавливаем предельные размеры для ограничивающих рамок
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # добавляем лица и рамки в соответствующие списки
            faces.append(build_face(frame, startY, endY, startX, endX))
            locs.append((startX, startY, endX, endY))

    # пытаемся распознать, если было обнаружено хотя бы 1 лицо
    if len(faces) > 0:
        # для более быстрого вывода будем обрабатывать все обнаруженные лица сразу вместо цикла for
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return locs, preds


# всегда получаем изображение
def detect_camera_image(vs, face_net, mask_net):
    while True:
        # ормируем окно с изображением с камеры и устанавлваем его размеры
        frame = vs.read()
        frame = imutils.resize(frame, width=600, height=800)

        # обнаруживаем лица в кадре и определяем, носят они маски или нет
        (locs, preds) = detect_mask(frame, face_net, mask_net)

        # цикл с обнаружением лиц и их положением
        for (box, pred) in zip(locs, preds):
            # распаковываем распознанные данные
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # цвет надписи для ситуаций есть маска/нет маски
            color = GREEN if mask > withoutMask else RED

            # формируем надпись в рамке
            label = build_label(mask, withoutMask)

            # отображение текста и рамки в окне
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # показываем окошко с камерой
        cv2.imshow("Camera window", frame)
        key = cv2.waitKey(1) & 0xFF

        if stop_detection(key):
            break


if __name__ == '__main__':
    # загружаем сериализованную модель детектора лиц с диска
    prototxtPath = os.path.join(os.getcwd(), 'face_detector', 'deploy.prototxt')
    weightsPath = os.path.join(os.getcwd(), 'face_detector', 'res10_300x300_ssd_iter_140000.caffemodel')
    face_net = cv2.dnn.readNet(prototxtPath, weightsPath)

    # загружаем модель для определения наличия маски на лице
    mask_net = load_model("mask_detector.model")

    # начинаем видео стрим
    print("[INFO] starting video stream...")
    video_stream = VideoStream(src=0).start()
    video_stream = cv2.flip(video_stream.frame, 1)

    detect_camera_image(video_stream, face_net, mask_net)
    # останавливаем запущенные процессы
    cv2.destroyAllWindows()
    video_stream.stop()