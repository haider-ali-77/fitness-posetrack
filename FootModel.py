import tensorflow as tf
import numpy as np
import tensorflow.keras as keras


class FootModel:

    def __init__(self,modelpath,classes):
        self.model = self.resnet50(output_units=classes, l2_regularization_strength=0.04, inp_shape=(224, 224, 3),
                         weights=None, freeze_backbone=False, regularize_backbone=False)
        self.model.load_weights(modelpath)


    def predictfoot(self,img,bbox,margin=0):
      x1=bbox[0]
      x2=bbox[2]
      y1=bbox[1]
      y2=bbox[3]
      height=img.shape[0]
      weight=img.shape[1]
     # d = 20    #adding margins of 20 pixels
      x1 = x1 - margin
      x2 = x2 + margin
      y1 = y1 - margin
      y2 = y2 + margin
      if (x1>x2):
        temp=x1
        x1=x2
        x2=temp
      if (y1>y2):
        temp=y1
        y1=y2
        y2=temp
      if (y2>=height):
        y2=height-1
      if (x2>=weight):
        x2=weight-1
      if (y1>=height):
        y1=height-1
      if (x1>=weight):
        x1=weight-1
      if (x1<=0):
        x1=0
      if (y1<=0):
        y1=0


      img1=img[y1:y2, x1:x2]
      img_i=tf.image.resize_with_pad(img1, 224, 224)
      img_i = (img_i / 127.5) - 1
      expand_img = tf.expand_dims(img_i, axis=0)
      predictions=self.model.predict(expand_img)
      if tf.argmax(predictions[0]) in [1,2] and tf.math.reduce_max(predictions[0])< 0.85:
        return tf.constant(0)
      return tf.argmax(predictions[0])

    def resnet50(self,output_units=4, l2_regularization_strength=0.001, inp_shape=(224, 224, 3), weights='imagenet',
                 freeze_backbone=False,
                 regularize_backbone=True) -> object:
      l2_regularizer = keras.regularizers.l2(l2_regularization_strength)
      model = keras.applications.ResNet50(input_shape=inp_shape, weights=weights, include_top=False)
      if regularize_backbone and not freeze_backbone:
        model = add_regularization(model, l2_regularizer)
      if freeze_backbone:
        model.trainable = False
      features = model.output
      avgpool = keras.layers.GlobalAveragePooling2D()(features)
      output_layer = keras.layers.Dense(output_units, kernel_regularizer=l2_regularizer, activation="softmax")(avgpool)

      return keras.models.Model(inputs=model.input, outputs=output_layer)