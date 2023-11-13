# if __name__ == "__main__":
#         from ultralytics import YOLO
#         try:
#             model = YOLO("yolov8x.pt", task="detect")
#             model.train(
#                 **{'data': 'c://Users//admin//Desktop//Github//K-water_project//scripts//datasets//k_project_waffle_v1.0.0//exports//YOLO//data.yaml', 'epochs': 10, 'batch': 2, 'imgsz': [640, 640], 'lr0': 0.01, 'lrf': 0.01, 'rect': False, 'device': '0', 'workers': 2, 'seed': 0, 'verbose': True, 'project': 'hubs//k_project_v1.0.0', 'name': 'artifacts'}
#             )
#         except Exception as e:
#             print(e)
#             raise e


if __name__ == "__main__":
        from ultralytics import YOLO
        try:
            model = YOLO("yolov8x.pt", task="detect")
            model.train(
                **{'data': './datasets/k_project_waffle_v1.0.0/exports/YOLO/data.yaml', 'epochs': 10, 'batch': 2, 'imgsz': [640, 640], 'lr0': 0.01, 'lrf': 0.01, 'rect': False, 'device': '0', 'workers': 2, 'seed': 0, 'verbose': True, 'project': 'hubs//k_project_v1.0.0', 'name': 'artifacts'}
            )
        except Exception as e:
            print(e)
            raise e