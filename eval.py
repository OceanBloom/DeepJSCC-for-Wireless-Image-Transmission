import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import keras
import JSCC_Model
import matplotlib.pyplot as plt
import numpy as np

ratio = [0.09]
train_SNR = [1, 4, 7, 13, 19]
test_SNR = [1, 4, 7, 10, 13, 16, 19 ,22, 25]
lr = 1e-3
test_batch_size = 128
markers = ['*', '|', 'o', 's', 'd']
colors = ['#0072bd', '#d95319', '#edb120', '#7e2f8e', '#77ac30']
def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=255.0)

def eval(testX):
    testX = testX
    psnr_results = {tr_snr: [] for tr_snr in train_SNR}
    # 初始化为：{1: [], 4: [], 7: [], 13: [], 19: [] }
    for rt in ratio:
        for tr_snr in train_SNR:
            print('------------------------------loading model: ratio:{}, trainSNR:{}dB------------------------------'.format(rt, tr_snr))
            for te_snr in test_SNR:
                test_model = JSCC_Model.JSCC(rt, te_snr)
                # Build model
                input = keras.Input(shape=(32, 32, 3))
                output = test_model(input)
                test_model = keras.Model(input, output)
                test_model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=keras.losses.mean_squared_error, metrics=[psnr])
                test_model.load_weights(f'./checkpoint/JSCC_ratio{rt}_trainSNR{tr_snr}.weights.h5')
                test_loss, test_psnr = test_model.evaluate(testX, testX, batch_size=test_batch_size, verbose=1)
                print('\n  testing: testSNR:{}dB, test_loss:{}, test_psnr:{}dB \n'.format(te_snr, test_loss, test_psnr))
                psnr_results[tr_snr].append(test_psnr)
    return psnr_results

def visualize(testX):
    # 随机选取一张原始图片
    random_index = np.random.randint(0, len(testX))
    original_image = testX[random_index]    
    # 创建 6x10 的子图布局
    fig, axes = plt.subplots(6, 10, figsize=(18, 10))
    # 标注第一行
    axes[0, 0].imshow(original_image.astype('uint8'))
    axes[0, 0].axis('off')
    for j in range(1, 10):
        axes[0, j].text(0.5, 0.5, f'Test SNR: {test_SNR[j - 1]}dB', horizontalalignment='center', verticalalignment='center')
        axes[0, j].axis('off')
    # 标注第一列
    for i in range(1, 6):
        axes[i, 0].text(0.5, 0.5, f'Train SNR: {train_SNR[i - 1]}dB', horizontalalignment='center', verticalalignment='center')
        axes[i, 0].axis('off')
    # 填充剩余的接收端恢复图
    for rt in ratio:
        for i, tr_snr in enumerate(train_SNR):
            for j, te_snr in enumerate(test_SNR):
                test_model = JSCC_Model.JSCC(rt, te_snr)
                # Build model
                input = keras.Input(shape=(32, 32, 3))
                output = test_model(input)
                test_model = keras.Model(input, output)
                test_model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=keras.losses.mean_squared_error, metrics=[psnr])
                test_model.load_weights(f'./checkpoint/JSCC_ratio{rt}_trainSNR{tr_snr}.weights.h5')
                input_image = np.expand_dims(original_image, axis=0)
                restored_image = test_model.predict(input_image)[0]
                axes[i + 1, j + 1].imshow(restored_image.astype('uint8'))
                axes[i + 1, j + 1].axis('off')
    fig.suptitle('AWGN channel visualization (k/n=1/12)')
    plt.show()



if __name__ == '__main__':
    # 获取测试集
    (_, _), (testX, _) = keras.datasets.cifar10.load_data()
    testX = testX.astype('float32')
    # PSNR曲线
    psnr_results = eval(testX)
    plt.figure(figsize=(8, 6))
    for i, tr_snr in enumerate(train_SNR):
        plt.plot(test_SNR, psnr_results[tr_snr], marker=markers[i], color=colors[i], label=f"trainSNR:{tr_snr} dB")
    plt.xlabel('SNR_test (dB)')
    plt.ylabel('PSNR (dB)')
    plt.title('AWGN channel(k/n=1/12)')
    plt.legend()
    plt.xlim(0, 25)
    plt.grid()
    plt.show()
    # 图像可视化
    visualize(testX)
    

