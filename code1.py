!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="BhJ63IemG3ACRXXxUyzC")
project = rf.workspace("dental-decay").project("check-hfbzl-zemwx")
version = project.version(18)
dataset = version.download("yolov8")

!pip install ultralytics

from ultralytics import YOLO

dataset_path = '/content/CHECK-18'

model = YOLO('yolov8n.pt')
train_params = {
    'data': '/content/CHECK-18/data.yaml',
    'epochs': 50,
    'batch_size': 16,
    'img_size': 640,
    'weights': 'yolov8n.pt',
    'hyp': {1.0, 1.5, 2.0},
}

model.train(data="/content/CHECK-18/data.yaml")

results = model.val()
print(results)

img = '/content/CHECK-18/train/images/1036_jpg.rf.ed4e791e759a8d6b8d18806a4370515d.jpg'

results = model.predict(img,save=True , conf=0.30)
results.show()