import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
import JSCC_Model


ratio = [0.09]
train_SNR = [1, 4, 7, 13, 19]
test_SNR = [1, 4, 7, 10, 13, 16, 19 ,22, 25]
batch_size = 64
epochs = 1000
lr = 1e-3
def train():
    #trainX shape:(50000, 32, 32, 3)  testX shape: (10000, 32, 32, 3)
    (trainX, _), (_, _) = keras.datasets.cifar10.load_data()
    trainX = trainX.astype('float32')
    for rt in ratio:
        for snr in train_SNR:
            print('***************ratio:{}, SNR:{}dB***************'.format(rt, snr))

            model = JSCC_Model.JSCC(rt, snr)
            # Build model
            input = keras.Input(shape=(32, 32, 3))
            output = model(input)
            model = keras.Model(input, output)
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=keras.losses.mean_squared_error)
            save_func = keras.callbacks.ModelCheckpoint(filepath=f'./checkpoint/JSCC_ratio{rt}_trainSNR{snr}.weights.h5', save_weights_only=True)
            model.fit(x=trainX, y=trainX, epochs=epochs, verbose=1, batch_size=batch_size, callbacks=[save_func])


if __name__ == '__main__':
    train()     

