from keras.preprocessing.image import ImageDataGenerator

image_dim = 24
i = 0

train_path = r"C:\Users\arkha\OneDrive\Desktop\ML\Deep Learning\Sign Detection\DataSet\Train"
    
train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.5,
        zoom_range = 0.5,
        rotation_range = 40,
        horizontal_flip=True)

for batch in train_datagen.flow_from_directory(
        train_path,
        target_size = (image_dim,image_dim),
        batch_size = 5,
        classes=['Ball', 'Left','Return','Right','Stop'],
        save_to_dir = r"C:\Users\arkha\OneDrive\Desktop\ML\Deep Learning\Sign Detection\Augmented_Dataset",
        save_prefix = "img",
        save_format = "jpeg"):
    i += 1
    if (i >= 100):
        break



    
