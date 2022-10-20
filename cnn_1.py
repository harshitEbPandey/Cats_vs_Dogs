import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image_dataset_from_directory

image_size = (244, 244)
batch_size = 16

train_generator = tf.keras.preprocessing.image_dataset_from_directory(
    "petimages",
    validation_split=0.2,
    labels='inferred',
    label_mode='binary',
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    shuffle = True,
)

validation_generator = tf.keras.preprocessing.image_dataset_from_directory(
    "petimages",
    validation_split=0.2,
    subset="validation",
    labels='inferred',
    label_mode='binary',
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    shuffle = True,
)

test_generator = validation_generator.take(20)
validation_generator = validation_generator.skip(20)

INPUT_SHAPE = (None, 244, 244, 3)

model = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(1/255.),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=RMSprop(learning_rate=0.0001), loss='binary_crossentropy', 
              metrics=['accuracy', 
                       tf.keras.metrics.Precision(name="precision"),
                       tf.keras.metrics.Recall(name="recall"),
                       tf.keras.metrics.TrueNegatives(name="tn"),
                       tf.keras.metrics.TruePositives(name="tp"),
                       tf.keras.metrics.FalseNegatives(name="fn"),
                       tf.keras.metrics.FalsePositives(name="fp")
                       ]
              )



model.build(INPUT_SHAPE)
#model.summary()

history = model.fit(train_generator,
                              epochs=2,
                              verbose=1,
                              validation_data=validation_generator)

loss, accuracy, precision, recall, tn, tp, fn, fp = model.evaluate(test_generator, verbose=1,batch_size=16)
print("loss - ", loss)
print("accuracy - ", accuracy)
print("precision - ", precision)
print("recall - ", recall)




