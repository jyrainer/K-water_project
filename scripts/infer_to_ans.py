from waffle_utils.image.io import load_image, save_image
from waffle_utils.file import io, network
inference_json_path = '/home/jyp/waffle/hubs/k_project_v1.0.0/inferences/inferences.json'
answer_json_path = '/home/jyp/waffle/datasets/k_project/labels/answer_sample.json'


inference_json = io.load_json(inference_json_path)
answer_json = io.load_json(answer_json_path)


# 키 : 이미지 파일 이름, 값 : 이미지 아이디
images_id_filename_dict = {}
anns_dict = {}
anns_list = []
# 삽입
for answer_id in answer_json["images"] :
    k, v = answer_id['file_name'], answer_id['id']
    images_id_filename_dict[k] = v

anns_count = 0
# inference.json 순회
for infer in inference_json :
    if list(infer.values())[0] == []:
        continue        # 답 없을때 그냥 컨티뉴
    else:
        for anns in list(infer.values())[0]:
            anns_id = int(anns_count)
            image_id = int(images_id_filename_dict[list(infer.keys())[0]])
            category_id = int(anns["category_id"])
            area = float(anns['bbox'][2]*anns['bbox'][3])
            bbox = anns['bbox']
            anns_dict = {
                "id": anns_id,
                "image_id": image_id,
                "category_id": category_id - 1,
                "segmentation": [],
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
                "attributes": {
                    "occluded": None,
                    "rotation": 0.0
                }
            }
            anns_list.append(anns_dict)
            anns_count += 1
            
            
answer_json['annotations'] = anns_list

io.save_json(answer_json,'./answer.json')