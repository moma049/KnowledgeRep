import gradio as gr
import tensorflow as tf
import cv2
import numpy 
from PIL import ImageTk, Image
model = tf.keras.models.load_model('my_model.h5')
classes = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing veh over 3.5 tons',
    11: 'Right-of-way at intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Veh > 3.5 tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve left',
    20: 'Dangerous curve right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End speed + passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End no passing vehicle with a weight greater than 3.5 tons'
}
def road_sign(img):

    if img is not None:
        print(img)
    
        image = Image.open(img)
        image = image.convert("RGB")
        image = image.resize((30,30))
        image = numpy.array(image) / 255.0
        image = numpy.expand_dims(image, axis=0)
        image = numpy.array(image)

        prediction = model.predict(image)
        pred = numpy.argmax(prediction, axis=1)
        return {str(classes[i]): float(prediction[0][i]) for i in range(43)}
    else: 
        return ' '
    

iface = gr.Interface(
    fn = road_sign,
    inputs = gr.Image(type="filepath", image_mode='RGB', sources='upload'),
    outputs = gr.Label(num_top_classes=3),
    live=True
)
iface.launch() 