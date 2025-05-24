# Import Kivy dependencies
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

# Import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Import other dependencies
import cv2
import tensorflow as tf
from Layer import L1Dist
import os
import numpy as np

# Build app layout
class FaceID(App):

    def build(self):
        self.camera = Image(size_hint=(1,.8))
        self.button = Button(text="Verify", on_press = self.verify, size_hint=(1,.1))
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1,.1))

        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.camera)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # Load tensorflow/keras model
        self.model = tf.keras.models.load_model('app/siamesemodel.h5', custom_objects={'L1Dist':L1Dist})

        # Capture 
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)

        return layout
    
    def update(self, *args):

        # Read frame from opencv
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]

        # Flip horizontall and convert image to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.camera.texture = img_texture

    def Preprocess(self, file_path):
        # Read in the image / Pročitaj sliku
        byte_img = tf.io.read_file(file_path)
        # Load in the image / Učitaj sliku  
        img = tf.io.decode_jpeg(byte_img)
    
        # Preprocessing steps / Koraci obrade
        img = tf.image.resize(img, (100, 100))
        # Scale the image to [0,1] / Promijeni veličinu slike na [0,1]
        img = img / 255.0

        # Return image
        return img
    
    def verify(self, *args):
        detection_threshold = 0.8
        verification_threshold = 0.5

        SAVE_PATH = os.path.join('app','application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        cv2.imwrite(SAVE_PATH, frame)

        results = []    
        for image in os.listdir(os.path.join('app','application_data', 'verification_images')):
            input_img = self.Preprocess(os.path.join('app','application_data', 'input_image', 'input_image.jpg'))
            validation_img = self.Preprocess(os.path.join('app','application_data', 'verification_images', image))

            # Make Predictions 
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            # result = model.predict([np.expand_dims(input_img, axis=0), np.expand_dims(validation_img, axis=0)])
            results.append(result)

        # Detection Threshold: Metric above which a prediciton is considered positive 
        detection = np.sum(np.array(results) > detection_threshold)
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
    
        # Verification Threshold: Proportion of positive predictions / total positive samples     
        verified = verification > verification_threshold

        # Set verification text
        self.verification_label.text = 'Verified' if verified == True else 'Unverified'

        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)

        return results, verified
        
    
if __name__ == '__main__':
    FaceID().run()