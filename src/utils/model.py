import tensorflow as tf 



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