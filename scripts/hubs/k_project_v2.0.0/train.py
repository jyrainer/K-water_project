if __name__ == "__main__":
        from ultralytics import YOLO
        try:
            model = YOLO("/home/jyp/waffle/hubs/k_project_v1.2.0/weights/best_ckpt_25_backup.pt", task="detect")
            model.train(
                **{'data': '/home/jyp/waffle/datasets/k_project_waffle_v1.0.0/exports/YOLO/data.yaml', 'epochs': 200, 'batch': 64, 'imgsz': [640, 640], 'lr0': 0.01, 'lrf': 0.01, 'rect': False, 'device': '7', 'workers': 2, 'seed': 0, 'verbose': True, 'project': '/home/jyp/waffle/hubs/k_project_v2.0.0', 'name': 'artifacts', 'patience': 200}
            )
        except Exception as e:
            print(e)
            raise e
        