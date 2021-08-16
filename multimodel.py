
import argparse
import numpy as np
import pandas as pd
import tensorflow.keras
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications import NASNetLarge
#from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt
import tensorflow as tf
#import horovod.tensorflow.keras as hvd


from data import SampleDataGenerator, make_dataset

def learning_plot(title, hist):
    plt.figure(figsize = (18,6))
    # accuracy
    plt.subplot(1, 2, 1)
    plt.plot(hist.history["accuracy"], label = "acc", marker = "o")
    plt.plot(hist.history["val_accuracy"], label = "val_acc", marker = "o")
    #plt.yticks(np.arange())
    #plt.xticks(np.arange())
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.title(title)
    plt.legend(loc = "best")
    plt.grid(color = 'gray', alpha = 0.2)
    # loss
    plt.subplot(1, 2, 2)
    plt.plot(hist.history["loss"], label = "loss", marker = "o")
    plt.plot(hist.history["val_loss"], label = "val_loss", marker = "o")
    #plt.yticks(np.arange())
    #plt.xticks(np.arange())
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title(title)
    plt.legend(loc = "best")
    plt.grid(color = "gray", alpha = 0.2)
    plt.show()
    plt.savefig(title + ".png")

def build_model(num_classes):
    #tf.debugging.set_log_device_placement(True)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        #マイクロアレイモデル作成
        input_microarray = Input(shape=(4835,), name="microarray")
        dense1 = Dense(4096, activation='tanh')(input_microarray)
        dropout1 = Dropout(0.5)(dense1)
        dense2 = Dense(2048, activation='tanh')(dropout1)
        dropout2 =Dropout(0.5)(dense2)
        dense3 = Dense(2048, activation='tanh')(dropout2)
        dropout3 = Dropout(0.5)(dense3)
        dense4 = Dense(2048, activation='tanh')(dropout3)
        dropout4 = Dropout(0.5)(dense4)
        dense5 = Dense(1024, activation='tanh')(dropout4)

        microarray_model = Model(inputs=input_microarray, outputs=dense5)

        #画像モデル作成
        model_name = "InceptionResNetV2"
        input_tensor = Input(shape=(512, 512, 3),name="img")
        base_model = InceptionResNetV2(
                input_tensor=input_tensor,
                include_top = False,
                weights = "imagenet",
                input_shape = None)
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='tanh')(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation = 'tanh')(x)
        image_model = Model(inputs=base_model.input, outputs=x)

        # # 249層までfreeze
        for layer in image_model.layers[:249]:
            layer.trainable = False
            if layer.name.startswith('batch_normalization'):
                 layer.trainable = True
        # # 250層以降再学習
        for layer in image_model.layers[249:]:
             layer.trainable = True


        combined = concatenate([image_model.output, microarray_model.output])
        z = Dense(512, activation='relu')(combined)
        #z = Dense(10, activation='sigmoid')(z)
        z = Dense(2, activation='linear')(z)
        model = Model(inputs=[base_model.input, microarray_model.input], outputs=z)

    return model

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-6, metavar='LR',
                        help='learning rate (default: 1.0)')

    args = parser.parse_args()


    # Horovod: initialize Horovod.
    #hvd.init()

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    K.set_session(tf.Session(config=config))
    """

    #df = pd.read_csv('/groups2/gcc50492/images_repeatliver/tile_microarray_finding_top10_train.csv', index_col=0)
    #df_test = pd.read_csv('/groups2/gcc50492/images_repeatliver/tile_microarray_finding_top10_test.csv', index_col=0)
    #df_microarray = pd.read_csv('/groups2/gcc50492/images_repeatliver/repeatliver_myegene_dropna_normalized.csv')
    #df_microarray = pd.read_csv('/groups2/gcc50492/images_repeatliver/repeatliver_myegene_dropna_normalized.csv')
    df = pd.read_csv('small_train.csv', index_col=0)
    df_test = pd.read_csv('small_train.csv', index_col=0)
    df_microarray = pd.read_csv('repeatliver_myegene_dropna_normalized.csv',header=0)

    df = df.sample(frac=1).reset_index(drop=True)
    df_train, df_valid = train_test_split(df, stratify=df['FINDING_TYPE_x'])

    epochs = args.epochs

    #マイクロアレイデータの正規化
    """
    def normalize_microarray(df):
        mean = df.iloc[:,1:].stack().mean()
        std = df.iloc[:,1:].stack().std(ddof=1)

        df_norm_value = (df.iloc[:,1:]-mean)/std

        df_normalized = pd.concat([df.iloc[:,0], df_norm_value], axis=1)

        return df_normalized

    df_microarray = normalize_microarray(df_microarray)
    """



    #画像+マイクロアレイデータ読みこみ

    batch_size = 2
    num_classes = 2
    train_generator = SampleDataGenerator(df_train, df_microarray, batch_size, num_classes,  shuffle=True)
    valid_generator = SampleDataGenerator(df_valid, df_microarray, batch_size, num_classes, shuffle=False)
    test_generator  = SampleDataGenerator(df_test, df_microarray, batch_size, num_classes, shuffle=False)

    # Horovod: adjust learning rate based on number of GPUs.
    #scaled_lr = args.lr * hvd.size()
    scaled_lr = args.lr
    opt = tensorflow.keras.optimizers.Adam(learning_rate=scaled_lr)

    # Horovod: add Horovod Distributed Optimizer.
    #opt = hvd.DistributedOptimizer(opt)

    model=build_model(num_classes)
    model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])
    #model.summary()
    #model = multi_gpu_model(model, gpus=8) 
    #model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10, verbose = 1)

    """
    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),

        # Horovod: average metrics among workers at the end of every epoch.
        #
        # Note: This callback must be in the list before the ReduceLROnPlateau,
        # TensorBoard or other metrics-based callbacks.
        hvd.callbacks.MetricAverageCallback(),

        # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
        # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
        hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, initial_lr=scaled_lr, verbose=1),

        # Reduce the learning rate if training plateaues.
        tensorflow.keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1), early_stopping
    ]
    #if  hvd.rank() == 0:
    #    callbacks.append(tensorflow.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

    hist = model.fit_generator(train_generator,
                        steps_per_epoch=(len(df_train)//8)//hvd.size(),
                        epochs=epochs,
                        verbose=1,
                        validation_data=validation_generator,
                        shuffle=False,
                        validation_steps=3*((len(df_val)//8)//hvd.size()),
                        class_weight=weight,
                        callbacks=callbacks)

    """

    callbacks = [
        #tensorflow.keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1),
        early_stopping,
        tensorflow.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5',save_best_only=True,verbose=1),
        ]

    """
    #hist = model.fit_generator(train_generator,
    hist = model.fit(train_generator,
                        epochs=epochs,
                        verbose=1,
                        validation_data=validation_generator,
                        shuffle=False,
                        class_weight=weight,
                        callbacks=callbacks)
    """
    hist = model.fit(x=make_dataset(train_generator),
            validation_data=make_dataset(valid_generator),
            steps_per_epoch=(len(df_train)//batch_size),
            validation_steps=(len(df_valid)//batch_size),
            epochs=10,
            class_weight=train_generator.class_weight,
            verbose=1,
            callbacks=callbacks,
            workers=4,
            use_multiprocessing=True,
            )

    learning_plot('inceptionresnetv3_0930_opt=adam_lr=1e-6', hist)
    #model.save('/groups2/gcc50492/images_repeatliver/models/model_inceptionresnetv3.hdf5')


    score = model.evaluate_generator(test_generator,
                steps=(len(df_train)//batch_size),
                verbose = 1)
    print("evaluate loss: {[0]:.4f}".format(score))
    print("evaluate acc: {[1]:.1%}".format(score))

    #model.save('/groups2/gcc50492/images_repeatliver/models/model_inceptionresnetv3.hdf5')


if __name__ == "__main__":
    main()

