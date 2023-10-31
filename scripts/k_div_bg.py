from waffle_utils.file import io, network
import os

train_json = io.load_json('/home/jyp/waffle/datasets/k_project/labels/train.json')
img_dict = {}
img_id_set = set()
bg_result_path = '/home/jyp/waffle/datasets/k_project/bg'
ann_img_id_set = set()
ann_list = []

for anno in train_json["annotations"]:
    ann_img_id_set.add(anno["image_id"])
    ann_list.append(anno)

for index, image in enumerate(train_json["images"]):
    if not(image["id"] in ann_img_id_set) :
        src = f"/home/jyp/waffle/datasets/k_project/train/" + image["file_name"]
        dst = f"/home/jyp/waffle/datasets/k_project/bg/" + image["file_name"]       
        io.copy_file(src, dst)
    else :
        print(image)