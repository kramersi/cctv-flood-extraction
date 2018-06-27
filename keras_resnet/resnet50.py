from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras import layers
from keras import models
from keras import optimizers
from keras import callbacks
from keras.utils import to_categorical
from keras_unet.k_unet import load_images, load_masks
import numpy as np

import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt


from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator



IM_WIDTH, IM_HEIGHT = 224, 224  # fixed size for ResNet
NB_EPOCHS = 3
BAT_SIZE = 32
FC_SIZE = 1024
NB_LAYERS_TO_FREEZE = 172


def get_nb_files(directory):
  """Get number of files by searching directory recursively"""
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*")))
  return cnt


def add_new_last_layer(base_model, nb_classes):
    """Add last layer to the convnet
    Args:
    base_model: keras model excluding top
    nb_classes: # of classes
    Returns:
    new keras model with last layer
    """
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(FC_SIZE, activation='relu')(x)
    predictions = layers.Dense(nb_classes, activation='softmax')(x)
    model = models.Model(input=base_model.input, output=predictions)
    return model


def setup_to_transfer_learn(model, base_model):
    """ Freeze all layers and compile the model

    """
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


def setup_to_finetune(model):
    """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top
      layers.
    note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in
         the inceptionv3 architecture
    Args:
     model: keras model
    """
    for layer in model.layers[:NB_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_LAYERS_TO_FREEZE:]:
        layer.trainable = True

    model.compile(optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
                  loss='categorical_crossentropy', metrics=['accuracy'])

def train(self, model_dir, train_dir, valid_dir, epochs=20, batch_size=4, nb_classes=2):
    """ trains a unet instance on keras. With on-line data augmentation to diversify training samples in each batch.

        example of defining paths
        train_dir = "E:\\watson_for_trend\\3_select_for_labelling\\train_cityscape\\"
        model_dir = "E:\\watson_for_trend\\5_train\\cityscape_l5f64c3n8e20\\"

    """
    seed = 1234

    x_tr = load_images(os.path.join(train_dir, 'images', '0'))  # load training pictures in numpy array
    shape = x_tr.shape  # pic_nr x width x height x depth
    n_train = shape[0]  # len(image_generator)

    # define callbacks
    mc1 = callbacks.ModelCheckpoint(os.path.join(model_dir, 'tl_model.h5'), save_best_only=True, save_weights_only=False)
    mc2 = callbacks.ModelCheckpoint(os.path.join(model_dir, 'ft_model.h5'), save_best_only=True, save_weights_only=False)
    es = callbacks.EarlyStopping(patience=9)
    tb = callbacks.TensorBoard(log_dir=model_dir)

    y_tr = load_masks(os.path.join(train_dir, 'masks', '0'))  # load mask arrays
    x_va = load_images(os.path.join(valid_dir, 'images', '0'))
    y_va = load_masks(os.path.join(valid_dir, 'masks', '0'))
    n_valid = x_va.shape[0]

    x_tr = self.normalize(x_tr)
    x_va = self.normalize(x_va)

    # create one-hot
    y_tr = to_categorical(y_tr, self.n_class)
    y_va = to_categorical(y_va, self.n_class)

    image_datagen = ImageDataGenerator(featurewise_center=False,
                                       featurewise_std_normalization=False,
                                       width_shift_range=0.0,
                                       height_shift_range=0.0,
                                       horizontal_flip=True,
                                       zoom_range=0.0)

    # calculate mean and stddeviation of training sample for normalisation (if featurwise center is true)
    # image_datagen.fit(x_tr, seed=seed)

    # create image generator for online data augmentation
    train_generator = image_datagen.flow(x_tr, y_tr, batch_size=batch_size, shuffle=True, seed=seed)
    valid_generator = (x_va, y_va)

    # setup model
    base_model = ResNet50(weights='imagenet', include_top=False)  # include_top=False excludes final FC layer
    model = add_new_last_layer(base_model, nb_classes)

    # transfer learning
    setup_to_transfer_learn(model, base_model)

    history_tl = model.fit_generator(train_generator,
                        validation_data=valid_generator,
                        steps_per_epoch=n_train / batch_size,
                        validation_steps=n_valid / batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=[mc1, es, tb],
                        use_multiprocessing=False,
                        workers=4)

    # fine-tuning
    setup_to_finetune(model)

    history_ft = model.fit_generator(train_generator,
                        validation_data=valid_generator,
                        steps_per_epoch=n_train / batch_size,
                        validation_steps=n_valid / batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=[mc2, es, tb],
                        use_multiprocessing=False,
                        workers=4)

    model.save(args.output_model_file)
    plot_training(history_ft)
    scores = self.model.evaluate(x_va, y_va, verbose=1)
    print('scores', scores)

def train(train_dir, val_dir, nb_epoch, nb_classes, batch_size):
    """Use transfer learning and fine-tuning to train a network on a new dataset"""
    nb_train_samples = get_nb_files(train_dir)
    nb_val_samples = get_nb_files(val_dir)
    nb_epoch = int(nb_epoch)
    batch_size = int(batch_size)

    # data prep
    train_datagen =  ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_directory(
        args.train_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
    )

    validation_generator = test_datagen.flow_from_directory(
        args.val_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
    )

    # setup model
    base_model = ResNet50(weights='imagenet', incluede_top=False)  # include_top=False excludes final FC layer
    model = add_new_last_layer(base_model, nb_classes)

    # transfer learning
    setup_to_transfer_learn(model, base_model)

    history_tl = model.fit_generator(
        train_generator,
        nb_epoch=nb_epoch,
        samples_per_epoch=nb_train_samples,
        validation_data=validation_generator,
        nb_val_samples=nb_val_samples,
        class_weight='auto')

    # fine-tuning
    setup_to_finetune(model)

    history_ft = model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_val_samples,
        class_weight='auto')

    model.save(args.output_model_file)

    if args.plot:
        plot_training(history_ft)


def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.show()


if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--train_dir")
    a.add_argument("--val_dir")
    a.add_argument("--nb_epoch", default=NB_EPOCHS)
    a.add_argument("--batch_size", default=BAT_SIZE)
    a.add_argument("--output_model_file", default="inceptionv3-ft.model")
    a.add_argument("--plot", action="store_true")

    args = a.parse_args()
    if args.train_dir is None or args.val_dir is None:
        a.print_help()
        sys.exit(1)

    if (not os.path.exists(args.train_dir)) or (not os.path.exists(args.val_dir)):
        print("directories do not exist")
        sys.exit(1)
    train_dir = 'E:'
    val_dir = 'df'
    train(val_dir, train_dir=train_dir, nb_epoch=3, batch_size=4, output_model_file)



    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print('Predicted:', decode_predictions(preds, top=3)[0])
    # Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]