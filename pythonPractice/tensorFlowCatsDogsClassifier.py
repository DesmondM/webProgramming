import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# 1. Set up paths (modify these to match your file locations)
dataset_dir = 'kagglecatsanddogs/PetImages'  # Update this path
train_dir = os.path.join(dataset_dir, 'train')
validation_dir = os.path.join(dataset_dir, 'validation')

# 2. Create train/validation directories (run once)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

# Create class subdirectories
for cls in ['Cat', 'Dog']:
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, cls), exist_ok=True)

# 3. Data Preparation (split into train/validation - run once)
# Note: Original dataset has Cat and Dog folders with images
def split_data(source, train, val, split_size=0.2):
    files = os.listdir(source)
    num_val = int(len(files) * split_size)
    
    for i, file in enumerate(files):
        src = os.path.join(source, file)
        if i < num_val:
            dst = os.path.join(val, file)
        else:
            dst = os.path.join(train, file)
        if os.path.getsize(src) > 0:  # Skip zero-byte files
            os.rename(src, dst)

# Run splitting (modify paths as needed)
split_data('kagglecatsanddogs/PetImages/Cat', 
           os.path.join(train_dir, 'Cat'), 
           os.path.join(validation_dir, 'Cat'))

split_data('kagglecatsanddogs/PetImages/Dog', 
           os.path.join(train_dir, 'Dog'), 
           os.path.join(validation_dir, 'Dog'))

# 4. Data Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),  # Resize images
    batch_size=32,
    class_mode='binary'      # Cat=0, Dog=1
)

validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# 5. Build CNN Model
model = models.Sequential([
    # First convolution block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    
    # Second convolution block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Third convolution block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Fourth convolution block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Classifier
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary output
])

model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
    metrics=['accuracy']
)

# 6. Train the Model
history = model.fit(
    train_generator,
    steps_per_epoch=100,  # 20000 images ÷ 32 batch size ≈ 625
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50   # 5000 images ÷ 32 ≈ 156
)

# 7. Save the Model
model.save('cats_vs_dogs_cnn.h5')

# 8. Evaluate Performance
# Plot training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.savefig('training_history.png')
plt.show()

# 9. Make Predictions on New Images
def predict_image(image_path):
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(150, 150)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    
    prediction = model.predict(img_array)
    if prediction < 0.5:
        return f"Cat ({100*(1-prediction[0][0]):.2f}% confidence)"
    else:
        return f"Dog ({100*prediction[0][0]:.2f}% confidence)"

# Test prediction
print('  ---- Test image is ------')
print(predict_image('images/testing.jpg'))