import tensorflow as tf 
import time
import os


def create_model(LOSS_FUNCTION,OPTIMIZER,METRICS,NUM_CLASSES):
    LAYERS=[
        tf.keras.layers.Flatten(input_shape=[28,28],name="InputLayer"),
        tf.keras.layers.Dense(300,activation="relu",name="HiddenLayer1"),
        tf.keras.layers.Dense(100,activation="relu",name="HiddenLayer2"),
        tf.keras.layers.Dense(NUM_CLASSES,activation="softmax",name="OutputLayer")
         ]
    model_clf=tf.keras.models.Sequential(LAYERS)
    model_clf.summary()

    model_clf.compile(optimizer=OPTIMIZER,loss=LOSS_FUNCTION,metrics=METRICS)

    return model_clf

def get_unique_filename(file_name):
    unique_filename=time.strftime(f"%Y%m%d_%H%M%S_{file_name}")
    return unique_filename

def save_model(model,model_name,model_dir):
    unique_filename=get_unique_filename(model_name)
    path_to_model=os.path.join(model_dir,unique_filename)
    model.save(path_to_model)
