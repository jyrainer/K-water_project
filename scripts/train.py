from ultralytics import YOLO
try:
    model = YOLO("yolov8n.pt", task="detect")
    model.train(
        data="/home/ljj/waffle/datasets/IsonDetDataset_v2.0.0/exports/YOLO/data.yaml",
        epochs=100,
        batch=256,
        imgsz=[640, 640],
        lr0=0.01,
        lrf=0.01,
        device="2",
        workers=16,
        project="IsonDet",
        name="v1.0.0",
        **{
            'patience':99,
            'scale':0.2,
            'mosaic':0.8,
           }
    )
except Exception as e:
    print(e)
    raise e
