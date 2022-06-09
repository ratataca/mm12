import glob
import os
from sqlite3 import Timestamp
import time

import cv2,io
import shutil
from flask import Blueprint, request, jsonify,send_file
import io
from base64 import encodebytes
from PIL import Image
import base64
import random
from PIL import Image 


data_bp = Blueprint('data', __name__, url_prefix='/data')

def make_timestame(timestamp):
    timestamp = timestamp.replace("-", "_")
    timestamp = timestamp.replace(":", "_")
    timestamp = timestamp.replace(" ", "_")
    return timestamp

def get_response_image(image_path):
    pil_img = Image.open(image_path, mode='r') # reads the PIL image
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img

@data_bp.route("/")
def data_index():
    timestamp = "2021-02-10 21:11:00"
    timestamp = make_timestame(timestamp)
    print(timestamp)
    return "Hello"

@data_bp.route("/video/upload", methods=["POST"])
def upload_video():
    if request.method == "POST":
        video = request.files["video"]
        user_id = request.form["user_id"]
        timestamp = request.form['timestamp']
        timestamp = make_timestame(timestamp)

        print(f"timestamp : {timestamp}")
        print("------")
        if video.filename == "":
            return jsonify(False)

        # if os.path.isdir(os.path.join(os.getenv("TEMP_DATA_PATH"), user_id)):
        #     shutil.rmtree(os.path.join(os.getenv("TEMP_DATA_PATH"), user_id))

        # 비디오 이미지 저장 경로
        video_path = os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, timestamp, "video")

        os.makedirs(video_path, exist_ok=True)
        os.makedirs(os.path.join(video_path, "JPEGImages"), exist_ok=True)
        os.makedirs(os.path.join(video_path, "Annotations"), exist_ok=True)
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
            cv2.imwrite(os.path.join(video_path, "JPEGImages", f"video-frame-{idx}.png"), frame)
            cv2.imwrite(os.path.join(video_path, "Annotations", f"video-frame-{idx}.png"), frame)
            idx += 1
        print("완료")

        tmp_list = {}
        for i in range(1,7):
            idx1 = random.randrange(1, idx-1)
            filename = f"JPEGImages/video-frame-{idx1}.png"
            with open(os.path.join(video_path, filename), 'rb') as img:
                base64_string1 = base64.b64encode(img.read()).decode("utf-8")
            tmp_list[str(i)] = base64_string1
        
        return jsonify(tmp_list)


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
        print("request : ",request.form)
        print("request : ",request.files)
        user_id = request.form["user_id"]
        timestamp = request.form['timestamp']
        timestamp = make_timestame(timestamp)
        current_page = int(request.form["current_page"])
        save_imgs1 = request.files["save_imgs1"]
        save_imgs2 = request.files["save_imgs2"]
        save_imgs3 = request.files["save_imgs3"]
        save_imgs4 = request.files["save_imgs4"]
        save_imgs5 = request.files["save_imgs5"]
        save_imgs6 = request.files["save_imgs6"]
        save_imgs7 = request.files["save_imgs7"]
        save_imgs8 = request.files["save_imgs8"]
        flag = request.form["flag"] # flag : true : 다음 페이지, false : 이전 페이지

        save_images_file = [save_imgs1,save_imgs2,save_imgs3,save_imgs4,save_imgs5,save_imgs6,save_imgs7,save_imgs8]
        # print("저장중")
        # idx=1
        # filename_pred = f"predict-frame-{idx}.png"
        # access_folder_path =  os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, timestamp, 'video')
        # access_pred_path = os.path.join(access_folder_path, "Predict", filename_pred)
        # cover_img = Image.open(save_imgs1.stream)
        # cover_img.save(access_pred_path)
        # print("저장완료")

        print()
        print("★★★ 받은 데이터 확인 ★★★")
        print()
        print("----------------------------------")
        s = f"user id : {user_id}\n" \
            f"current page : {current_page}\n" \
            f"flag : {flag}\n" \
            f"timestamp : {timestamp}"
        print(s)
        print("----------------------------------")
        # 유저 ID가 없다면 실패
        if user_id == "":
            return jsonify(False)

        # 비디오 이미지 저장 경로
        access_pred_path = os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, timestamp, 'video', "Predict")
        origin_imgs_path = os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, timestamp, 'video', "JPEGImages")
        access_folder_path =  os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, timestamp, 'video')

        total_length = len(glob.glob(access_pred_path + "/*.png"))
        print("접근할 경로 : ",access_pred_path)
        print("총 이미지 갯수 : ",total_length)
        print(access_folder_path)

        print("덮어쓰기 중   ...ing")
        ## 수정된 내역 데이터 저장 코드는 skip
        start_point,end_point = find_point(current_page)
        count_idx =0
        for idx in range(start_point,end_point):
            # 덮어쓸 파일 명
            filename_pred = f"predict-frame-{idx}.png"
            access_pred_path = os.path.join(access_folder_path, "Predict", filename_pred)

            # file = request.files['file']
            # save_imgs[idx].stream
            # print(save_imgs1[idx-1])
            cover_img = Image.open(save_images_file[count_idx].stream)
            cover_img.save(access_pred_path)
            count_idx +=1
            print("오잉 완료")
        print("덮어쓰기 완료")
        print()
        if flag:  # true이면 다음 페이지
            next_page = current_page +1
        else:  # false 이면 이전 페이지
            next_page = current_page -1

        start_point,end_point = find_point(next_page)
        result = {}  # 정보를 담을 변수
        origin_img_data = {}
        pred_img_data = {}
        # access_pred_path = os.path.join('fake_labe_folder','fake_image.png')
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



@data_bp.route("/video/testing", methods=["POST"])
def goto_test():
    if request.method == "POST":
        print("페이지 넘김 기능 Okay")
        # time.sleep(13)
        user_id = request.form["user_id"]
        timestamp = request.form['timestamp']
        timestamp = make_timestame(timestamp)

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
        access_folder_path =  os.path.join(os.getenv("TEMP_DATA_PATH"), user_id, timestamp, "video")

        start_point, end_point = 1,9
        origin_img_data = {}
        for idx in range(start_point,end_point):
            filename = f"video-frame-{idx}.png"
            access_origin_path = os.path.join(access_folder_path,"JPEGImages", filename)
            # Image를 base64로 변환하기
            with open(access_origin_path, 'rb') as img:
                base64_string = base64.b64encode(img.read()).decode("utf-8")
            origin_img_data[filename] = base64_string

        return jsonify(origin_img_data)