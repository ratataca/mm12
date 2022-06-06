import glob
import os
import shutil
import time

import numpy as np

import cv2
from flask import Blueprint, jsonify, request, send_file
from photovleml.service import PhotovleService

model_bp = Blueprint('model', __name__, url_prefix='/model')


@model_bp.route("/train", methods=["POST"])
def train():
    if request.method == "POST":
        img = request.files["img"]
        label = request.files["label"]
        user_id = request.form["user_id"]
        timestamp = request.form["timestamp"]
        
        if img.filename == "":
            return jsonify(False)

        if label.filename == "":
            return jsonify(False)

        img_path = os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, timestamp, "JPEGImages")
        label_path = os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, timestamp, "Annotations")

        if os.path.isdir(img_path):
            shutil.rmtree(img_path)

        if os.path.isdir(label_path):
            shutil.rmtree(label_path)

        os.makedirs(img_path, exist_ok=True)
        os.makedirs(label_path, exist_ok=True)

        img.save(os.path.join(img_path, img.filename))
        label.save(os.path.join(label_path, label.filename))

        PhotovleService.train(user_id=user_id, timestamp=timestamp)
        PhotovleService.predict_video(user_id=user_id, timestamp=timestamp)

        return jsonify(True)


@model_bp.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        img = request.files["img"]
        label = request.files["label"]
        user_id = request.form["user_id"]
        timestamp = request.form["timestamp"]

        if img.filename == "":
            return jsonify(False)

        if label.filename == "":
            return jsonify(False)

        img_path = os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, timestamp, "predict", "JPEGImages")
        label_path = os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, timestamp, "predict", "Annotations")


        if os.path.isdir(img_path):
            shutil.rmtree(img_path)

        if os.path.isdir(label_path):
            shutil.rmtree(label_path)

        os.makedirs(img_path, exist_ok=True)
        os.makedirs(label_path, exist_ok=True)

        img.save(os.path.join(img_path, img.filename))
        label.save(os.path.join(label_path, label.filename))

        return jsonify(PhotovleService.predict(user_id=user_id, timestamp=timestamp))

# 예측된 데이터 동영상 파일로 저장하기
@model_bp.route("/video", methods=["POST"])
def get_predicted_video():
    if request.method == "POST":
        user_id = str(request.json["user_id"])
        timestamp = request.form["timestamp"]
        PhotovleService.predict_video(user_id=user_id, timestamp=timestamp)

        return send_file(
            os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, timestamp, "output.avi"),
            # os.path.join(os.getenv("TEMP_DATA_PATH"), "video", "hand.mp4"),
            # attachment_filename='output.avi',
            # as_attachment=True,
            mimetype="video/x-msvideo"
        )

def mosaic(src, ratio=0.08):
    small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

# 예측된 데이터 동영상 파일로 저장하기
@model_bp.route("/video1", methods=["POST"])
def get_video():
    if request.method == "POST":
        # time.sleep(25)
        print("동영상 저장 중 .....")
        user_id = str(request.json["user_id"])
        timestamp = request.form["timestamp"]
        access_folder = os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, timestamp,"split_img")
        predict_folder = os.path.join('fake_labe_folder')

        total_count = len(glob.glob(access_folder+"/*.png"))
        os.makedirs(os.path.join(os.getenv("TEMP_DATA_PATH"), user_id,timestamp,"pred_img"),exist_ok=True)

        predict_img_list = []
        for idx in range(1, total_count+1):

            filename = f"video-frame-{idx}.png"
            mask_img = cv2.imread(os.path.join(predict_folder,"fake_image.png"),0)
            origin_img = cv2.imread(os.path.join(access_folder,filename))

            if idx == 1:
                W, H, C = origin_img.shape
                fcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                out = cv2.VideoWriter(os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, "output.avi"), fcc, 20, (H, W))

            # 임지 mask image
            mask_img = np.zeros((origin_img.shape[0], origin_img.shape[1], 3), np.uint8)
            cv2.circle(
                mask_img,
                (int(mask_img.shape[1] /4),
                 int(mask_img.shape[1] /4)),
                 int(mask_img.shape[1]/8),
                (255,255,255),
                -1)

            # 타켓 대상 추출
            bit_and_mask = cv2.bitwise_and(origin_img, mask_img)

            # 타켓 외 영역 추출출
            other_mask_img = np.where(mask_img == 0, 255, 0)
            other_mask_img = other_mask_img.astype('uint8')
            bit_and_other = cv2.bitwise_and(origin_img, other_mask_img)
            dst_01 = mosaic(bit_and_other, ratio=0.08)
            result_img = cv2.add(dst_01, bit_and_mask)
            cv2.imwrite(os.path.join(os.getenv("TEMP_DATA_PATH"), user_id,timestamp,"pred_img",filename), result_img)

            out.write(result_img)

        out.release()
        print("동영상 저장 완료")
        return send_file(
            os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, "output.avi"),
            mimetype="video/x-msvideo"
        )