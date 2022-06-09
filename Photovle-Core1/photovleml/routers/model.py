import glob
import os
import shutil
import time

import numpy as np

import cv2
from flask import Blueprint, jsonify, request, send_file
from photovleml.service import PhotovleService
import io
import base64

model_bp = Blueprint('model', __name__, url_prefix='/model')

def make_timestame(timestamp):
    timestamp = timestamp.replace("-", "_")
    timestamp = timestamp.replace(":", "_")
    timestamp = timestamp.replace(" ", "_")
    return timestamp

@model_bp.route("/train", methods=["POST"])
def train():
    if request.method == "POST":
        img = request.files["img"]
        label = request.files["label"]
        user_id = request.form["user_id"]
        timestamp = request.form["timestamp"]
        timestamp = make_timestame(timestamp)
        
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


        # 처음 8개 이미지 불로오기
        # 비디오 이미지 저장 경로
        access_folder_path =  os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, timestamp, 'video')

        start_point,end_point = 1,9
        result = {}  # 정보를 담을 변수
        origin_img_data = {}
        pred_img_data = {}
        for idx in range(start_point,end_point):
            filename = f"video-frame-{idx}.png"
            filename_pred = f"predict-frame-{idx}.png"
            access_origin_path = os.path.join(access_folder_path,"JPEGImages", filename)
            access_pred_path = os.path.join(access_folder_path, "Predict", filename_pred)  ## 추후 효진님 코드가 합쳐지면 예측 데이터 폴더에 맞게 합쳐질 예정
            print("접근한 경로 리스트 ")
            # print("파일 명 : ",filename)
            # Image를 base64로 변환하기
            with open(access_origin_path, 'rb') as img:
                base64_string = base64.b64encode(img.read()).decode("utf-8")

            label_img = cv2.imread(access_pred_path,0)
            label_img = label_img.tolist()
            pixels = []
            for y in range(len(label_img)):
                for x in range(len(label_img[y])):
                    if label_img[y][x] != 0:
                        pixels.append((x, y))

            origin_img_data[filename] = base64_string
            pred_img_data[filename] = pixels

        result['origin_data'] = origin_img_data
        result['label_data'] = pred_img_data
        
        return jsonify(result)
        # access_folder_path =  os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, timestamp, "video")
        # start_point, end_point = 1,9
        # origin_img_data = {}
        # for idx in range(start_point,end_point):
        #     filename = f"video-frame-{idx}.png"
        #     access_origin_path = os.path.join(access_folder_path,"JPEGImages", filename)
        #     # Image를 base64로 변환하기
        #     with open(access_origin_path, 'rb') as img:
        #         base64_string = base64.b64encode(img.read()).decode("utf-8")
        #     origin_img_data[filename] = base64_string

        # return jsonify(origin_img_data)
        # return jsonify(True)


@model_bp.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        img = request.files["img"]
        label = request.files["label"]
        user_id = request.form["user_id"]
        timestamp = request.form["timestamp"]
        timestamp = make_timestame(timestamp)

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
        timestamp = make_timestame(timestamp)
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
        timestamp = request.json["timestamp"]
        timestamp = make_timestame(timestamp)
        
        os.makedirs(os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, timestamp, "pred_img"),exist_ok=True)

        access_folder = os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, timestamp, 'video', "JPEGImages")
        predict_folder = os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, timestamp, "pred_img")

        total_count = len(glob.glob(access_folder+"/*.png"))

        predict_img_list = []
        for idx in range(1, total_count+1):

            filename = f"video-frame-{idx}.png"
            filename_pred = f"predict-frame-{idx}.png"
            result_filename = f"result-frame-{idx}.png"
            
            mask_img = cv2.imread(os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, timestamp, 'video', "Predict", filename_pred),1)
            origin_img = cv2.imread(os.path.join(access_folder,filename))

            if idx == 1:
                W, H, C = origin_img.shape
                fcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                out = cv2.VideoWriter(os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, timestamp, "output.avi"), fcc, 20, (H, W))

            # 타켓 대상 추출
            print("mask img shape",mask_img.shape)
            print("origin img shape",origin_img.shape)
            
            bit_and_mask = cv2.bitwise_and(origin_img, mask_img)

            # 원본 사진 모자이크 처리
            dst_01 = mosaic(origin_img)

            # 타켓 외 영역 마스크 영역 추출
            other_mask_img = np.where(mask_img == 0, 255, 0)
            other_mask_img = other_mask_img.astype('uint8')

            # 타켓 외 영역 원본 이미지 추출
            bit_and_other = cv2.bitwise_and(dst_01, other_mask_img)
            result_img = cv2.add(bit_and_other, bit_and_mask)
            cv2.imwrite(os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, timestamp,"pred_img",result_filename), result_img)

            out.write(result_img)

        out.release()
        print("동영상 저장 완료")
        return send_file(
            os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, timestamp, "output.avi"),
            mimetype="video/x-msvideo"
        )