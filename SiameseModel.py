# Import standard dependencies / Uvoz standardnih knjižica
import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt
import uuid

# Import tensorflow dependencies - Functional API / Uvoz tensorflow knjižica konkretno Functional API
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

# Avoid OOM (Out of memory) errors by setting GPU Memory Consumption Growth
# Sprječavanje OOM grešaka postavljanjem ograničenja za korištenje GPU memorije.
def GpuGrowth():   
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Move LFW images to the following repository data/negative / Premjesti LFW slike u direktorij data/negative
# def ReplacingLFWIntoNegativesFolder(): 
#     for directory in os.listdir('archive/lfw-deepfunneled/lfw-deepfunneled'):
#         for file in os.listdir(os.path.join('archive/lfw-deepfunneled/lfw-deepfunneled', directory)):
#             EX_PATH = os.path.join('archive/lfw-deepfunneled/lfw-deepfunneled', directory, file)
#             NEW_PATH = os.path.join(NEG_PATH, file)
#             os.replace(EX_PATH, NEW_PATH)

def Preprocess(file_path):
    
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

# Preprocess the images in the Dataset / Obradi slike u skupu podataka  
def Preprocess_Twin(input_img, validation_img, label):
    return(Preprocess(input_img), Preprocess(validation_img), label)

def make_embedding():
    inp = Input(shape=(100, 100, 3), name='input_image')

    # First block of convolutional layers / Prva blok konvolucijskih slojeva
    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2, 2), padding='same')(c1)   

    # Second block of convolutional layers / Druga blok konvolucijskih slojeva  
    c2 = Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2, 2), padding='same')(c2)

    # Third block of convolutional layers / Treća blok konvolucijskih slojeva
    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2, 2), padding='same')(c3)

    # Final embedding block / Završni blok  
    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)

    return Model(inputs=inp ,outputs=d1, name='Embedding')

# Siamise L1 distance Class / Siamise L1 udaljenost klasa
class L1Dist(Layer):

    # Init method / Inicijalizacija 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # Similarity method / Metoda sličnosti  
    def call(self, inputs):
        input_embedding, validation_embedding = inputs
        return tf.math.abs(input_embedding - validation_embedding)
    
def make_siamese_model():

    # Anchor input image in the network / Sidrova ulazna slika u mreži
    input_image = Input(name='input_image', shape=(100, 100, 3))

    # Validation input image in the network / Ulazna slika za validaciju u mreži
    validation_image = Input(name='validation_image', shape=(100, 100, 3))

    # Combine siamese distance components / Kombiniraj komponente siamiskih udaljenosti
    siamese_layer = L1Dist(name='distance')
    distances = siamese_layer([embedding(input_image), embedding(validation_image)])

    # Classification layer / Klasa klasifikacije    
    classifier = Dense(1, name='dense', activation='sigmoid')(distances)  

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

# GpuGrowth()
# # Setup paths / Konfiguracija putanja
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

# Make the directories / Stvori direktorije
# os.makedirs(POS_PATH)
# os.makedirs(NEG_PATH)
# os.makedirs(ANC_PATH)

# Establish the connection to the camera / Uspostavi vezu s kamerom 
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     ret, frame = cap.read()

#     Crop the frame to 250x250px / Isijeci sliku 
#     frame = frame[120:120+250, 200:200+250]  

#     Collect anchors images / Prikupi uzorke sidra
#     if cv2.waitKey(1) & 0xFF == ord('a'):
#         imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
#         cv2.imwrite(imgname, frame)

#     Collect positives images / Prikupi pozitivne uzorke 
#     if cv2.waitKey(1) & 0xFF == ord('p'):
#         imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
#         cv2.imwrite(imgname, frame)

#     Show image back to the screen / Prikaži sliku na ekranu   
#     cv2.imshow('Face recognition app', frame)   

#     # Exit the loop if 'q' is pressed / Izađi iz petlje ako je pritisnuta tipka 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

#     # Exit the loop if the window is closed / Izađi iz petlje ako je prozor zatvoren    
#     if cv2.getWindowProperty('Face recognition app', cv2.WND_PROP_VISIBLE) < 1:
#         break
# # Relase the camera and destroy all windows / Otpusti kameru i uništi sve prozore   
# cap.release()
# cv2.destroyAllWindows()

anchor = tf.data.Dataset.list_files(ANC_PATH +'\\*.jpg').take(300)
positive = tf.data.Dataset.list_files(POS_PATH +'\\*.jpg').take(300)
negative = tf.data.Dataset.list_files(NEG_PATH +'\\*.jpg').take(300)

# # dir_test = anchor.as_numpy_iterator() 

# # print(dir_test.next())

# # (anchor, positive) => 1,1,1,1,1
# # (anchor, negative) => 0,0,0,0,0

# # tf.zeros(len(anchor))

positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)  

# # samples = data.as_numpy_iterator() 

# # example = samples.next()    

# # Preprocess_Twin(*example)

# Bulid dataloader pipeline / Izgradi pipeline za učitavanje podataka   
data = data.map(Preprocess_Twin)
data = data.cache() 
data = data.shuffle(buffer_size=1024)

# Training partition / Particija za treniranje
train_data = data.take(round(len(data)*.7)) 
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)        

# Testing partition / Particija za testiranje   
test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16) 
test_data = test_data.prefetch(8)

# # Embedding model / Ugradbeni model
# embedding = make_embedding()
# siamese_model = make_siamese_model()
# siamese_model.summary()

# # Setup loss function and optimizer / Konfiguracija funkcije gubitka i optimizatora
# binary_cross_entropy = tf.keras.losses.BinaryCrossentropy()
# opt = tf.keras.optimizers.Adam(1e-4)

# # Establish the checkpoint directory / Uspostavi direktorij za checkpoint
# checkpoint_dir = './training_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
# checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

# @tf.function
# def train_step(batch):
    
#     with tf.GradientTape() as tape:
        
#         # Get anchor and positive/negative images / Uzimanje sidrenih i pozitivnih/negativnih slika
#         X = batch[:2]
#         # Get the labels / Uzimanje oznaka
#         Y = batch[2]

#         # Foward pass / Prolaz unaprijed
#         yhat = siamese_model(X, training=True)
#         # Calculate the loss / Izračunaj gubitak
#         loss = binary_cross_entropy(Y, yhat)

#     # Calculate gradients / Izračunaj gradijente
#     grad = tape.gradient(loss, siamese_model.trainable_variables)
    
#     # Calculate updated weights and apply to siamese model / Izračunaj ažurirane težine i primijeni na siamiskom modelu
#     opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
    
#     # Return loss / Vratiti gubitak
#     return loss

# # Training loop / Petlja treniranja
# def train(data, EPOCHS):
#     # Loop through epochs / Petlja kroz epohe
#     for epoch in range(1, EPOCHS+1):
#         print('\n Epoch {}/{}'.format(epoch, EPOCHS))
#         progbar = tf.keras.utils.Progbar(len(data))
        
#         # Loop through each batch / Petlja kroz svaki batch
#         for idx, batch in enumerate(data):
#             # Run train step here
#             train_step(batch)
#             progbar.update(idx+1)
        
#         # Save checkpoints / Spremi checkpoint
#         if epoch % 10 == 0: 
#             checkpoint.save(file_prefix=checkpoint_prefix)

# # Train the model / Treniraj model
# EPOCHS = 50 
# train(train_data, EPOCHS)

# Reload model 
model = tf.keras.models.load_model('app/siamesemodel.h5', 
                                   custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})
# # Import metric calcuations / Uvoz metrika
# from tensorflow.keras.metrics import Precision, Recall

# # Get a batch of test data / Uzimanje testnog skupa podataka    
# test_input, test_val, y_true = test_data.as_numpy_iterator().next()

# # Make predictions / Predikcija 
# y_hat = model.predict([test_input, test_val]) 

# # Post processing the results / Postprocesiranje rezultata
# y_pred = [1 if prediction > 0.5 else 0 for prediction in y_hat ]

# # Calculate the FAR and FRR / Izračunaj FAR i FRR  
# false_accepts = np.logical_and(y_pred == 1, y_true == 0)
# false_rejects = np.logical_and(y_pred == 0, y_true == 1)
# FAR = np.sum(false_accepts) / np.sum(y_true == 0)
# FRR = np.sum(false_rejects) / np.sum(y_true == 1)

# # Print the results / Ispis rezultata
# print(f"FAR: {FAR * 100:.2f}%")
# print(f"FRR: {FRR * 100:.2f}%")

# # Calculate precision and recall / Izračunaj preciznost i odziv
# m = Recall()

# # # Calculate the recall / Izračunaj odziv    
# m.update_state(y_true, y_hat)

# # # Return recall result / Vratiti rezultat odziva
# m.result().numpy()  

# # # Creating a metric object 
# m = Precision()

# # # Calculating the recall value 
# m.update_state(y_true, y_hat)

# # # Return Recall Result
# m.result().numpy()

# # Save the model / Spremi model
# # model.save('siamese_model.h5')

def verify(model, detection_threshold, verification_threshold):
    results = []    
    for image in os.listdir(os.path.join('application_data', 'verification_images')):
        input_img = Preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
        validation_img = Preprocess(os.path.join('application_data', 'verification_images', image))

        # Make Predictions 
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        # result = model.predict([np.expand_dims(input_img, axis=0), np.expand_dims(validation_img, axis=0)])
        results.append(result)

    # Detection Threshold: Metric above which a prediciton is considered positive 
    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
    
    # Verification Threshold: Proportion of positive predictions / total positive samples     
    verified = verification > verification_threshold

    return results, verified

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    frame = frame[120:120+250, 200:200+250] 

    cv2.imshow("Verification", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('v'):
        # snimanje slike i verifikacija
        cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)
        results, verified = verify(model, 0.9, 0.7)
        print(verified)
        print(results)
    elif key == ord('q'):
        break

    if cv2.getWindowProperty('Verification', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()