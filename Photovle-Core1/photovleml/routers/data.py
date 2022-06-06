import glob
import os
import time

import cv2,io
import shutil
from flask import Blueprint, request, jsonify,send_file
import io
from base64 import encodebytes
from PIL import Image
import base64

data_bp = Blueprint('data', __name__, url_prefix='/data')

def get_response_image(image_path):
    pil_img = Image.open(image_path, mode='r') # reads the PIL image
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img

@data_bp.route("/")
def data_index():
    return "Hello"

@data_bp.route("/video/upload", methods=["POST"])
def upload_video():
    if request.method == "POST":
        video = request.files["video"]
        user_id = request.form["user_id"]
        timestamp = request.form['timestamp']

        print(f"timestamp : {timestamp}")
        print("------")
        if video.filename == "":
            return jsonify(False)

        if os.path.isdir(os.path.join(os.getenv("TEMP_DATA_PATH"), user_id)):
            shutil.rmtree(os.path.join(os.getenv("TEMP_DATA_PATH"), user_id))

        # 비디오 이미지 저장 경로
        video_path = os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, "video")
        # 프레임마다 이미지 저장 경로
        frame_capture_path = os.path.join(os.getenv("TEMP_DATA_PATH"), user_id)

        os.makedirs(video_path, exist_ok=True)
        os.makedirs(os.path.join(frame_capture_path, "split_img"), exist_ok=True)
        os.makedirs(os.path.join(frame_capture_path, "predict_img"), exist_ok=True)
        print("비디오 저장하였습니다.")
        # 비디오 저장
        video.save(os.path.join(video_path, video.filename))

        # 비디오 불러와서 프레임별로 짜르기
        cap = cv2.VideoCapture(os.path.join(video_path, video.filename))
        idx = 1

        # 비디오 영상 프레임으로 짤라서 저장하기
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(frame_capture_path, "split_img", f"video-frame-{idx}.png"), frame)    # 원본 이미지 정보 경로
            cv2.imwrite(os.path.join(frame_capture_path, "predict_img", f"video-frame-{idx}.png"), frame)  # 마스크를 써야할 이미지 경로
            idx += 1
        print("완료")

        with open(os.path.join(frame_capture_path, f"split_img/video-frame-3.png"), 'rb') as img:
            base64_string1 = base64.b64encode(img.read()).decode("utf-8")
        with open(os.path.join(frame_capture_path, f"split_img/video-frame-10.png"), 'rb') as img:
            base64_string2 = base64.b64encode(img.read()).decode("utf-8")
        with open(os.path.join(frame_capture_path, f"split_img/video-frame-18.png"), 'rb') as img:
            base64_string3 = base64.b64encode(img.read()).decode("utf-8")
        with open(os.path.join(frame_capture_path, f"split_img/video-frame-28.png"), 'rb') as img:
            base64_string4 = base64.b64encode(img.read()).decode("utf-8")
        with open(os.path.join(frame_capture_path, f"split_img/video-frame-45.png"), 'rb') as img:
            base64_string5 = base64.b64encode(img.read()).decode("utf-8")
        with open(os.path.join(frame_capture_path, f"split_img/video-frame-80.png"), 'rb') as img:
            base64_string6 = base64.b64encode(img.read()).decode("utf-8")

        return jsonify({"1":base64_string1,
                        "2":base64_string2,
                        "3":base64_string3,
                        "4":base64_string4,
                        "5":base64_string5,
                        "6":base64_string6,
                        })

def find_point(n):
    # 1,2,3,4,5,6,7,8,9,10 -> 10개  1page
    # 11,12,...,20 -> 10개    2page
    # 21                      3page
    # page = n일때
    # start point = (n-1) * 8 +1
    # end point = start point + 10
    start_point = ((n - 1) * 8) + 1
    end_point = start_point + 8
    return start_point, end_point


@data_bp.route("/video/paging", methods=["POST"])
def goto_page():
    if request.method == "POST":
        print("페이지 넘김 기능 Okay")
        # time.sleep(10)
        user_id = request.form["user_id"]
        timestamp = request.form.get('timestamp',"000")
        current_page = int(request.form["current_page"])
        save_imgs = request.form["save_imgs"]
        flag = request.form["flag"] # flag : true : 다음 페이지, false : 이전 페이지

        print()
        print("★★★ 받은 데이터 확인 ★★★")
        print()
        print("----------------------------------")
        s = f"user id : {user_id}\n" \
            f"current page : {current_page}\n" \
            f"save imgs : {save_imgs}\n" \
            f"flag : {flag}\n" \
            f"timestamp : {timestamp}"
        print(s)
        print("----------------------------------")
        # 유저 ID가 없다면 실패
        if user_id == "":
            return jsonify(False)

        # 비디오 이미지 저장 경로
        access_pred_path = os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, "predict_img")
        origin_imgs_path = os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, "split_img")
        access_folder_path =  os.path.join(os.getenv("TEMP_DATA_PATH"), user_id)

        total_length = len(glob.glob(access_pred_path + "/*.png"))
        print("접근할 경로 : ",access_pred_path)
        print("총 이미지 갯수 : ",total_length)
        print(access_folder_path)
        ## 수정된 내역 데이터 저장 코드는 skip
        start_point,end_point = find_point(current_page)

        if flag:  # true이면 다음 페이지
            next_page = current_page +1
        else:  # false 이면 이전 페이지
            next_page = current_page -1

        start_point,end_point = find_point(next_page)
        result = {}  # 정보를 담을 변수
        origin_img_data = {}
        pred_img_data = {}
        access_pred_path = os.path.join('fake_labe_folder','fake_image.png')
        for idx in range(start_point,end_point):
            filename = f"video-frame-{idx}.png"
            access_origin_path = os.path.join(access_folder_path,"split_img", filename)
            # access_pred_path = os.path.join(access_folder_path, "predict_img", filename)  ## 추후 효진님 코드가 합쳐지면 예측 데이터 폴더에 맞게 합쳐질 예정
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



@data_bp.route("/video/testing", methods=["POST"])
def goto_test():
    if request.method == "POST":
        print("페이지 넘김 기능 Okay")
        # time.sleep(13)
        user_id = request.form["user_id"]

        print("★★★ 받은 데이터 확인 ★★★")
        print()
        print("----------------------------------")
        s = f"user id : {user_id}\n"
        print(s)
        print("----------------------------------")
        # 유저 ID가 없다면 실패
        if user_id == "":
            return jsonify(False)

        # 비디오 이미지 저장 경로
        access_folder_path =  os.path.join(os.getenv("TEMP_DATA_PATH"), user_id)

        start_point, end_point = 1,9
        origin_img_data = {}
        for idx in range(start_point,end_point):
            filename = f"video-frame-{idx}.png"
            access_origin_path = os.path.join(access_folder_path,"split_img", filename)
            # Image를 base64로 변환하기
            with open(access_origin_path, 'rb') as img:
                base64_string = base64.b64encode(img.read()).decode("utf-8")
            origin_img_data[filename] = base64_string

        return jsonify(origin_img_data)