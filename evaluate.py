from Tiramisu import Tiramisu
import numpy as np
import argparse
import os

def calculate_iou(nb_classes, labels, predictions):
    conf_m = np.zeros((nb_classes, nb_classes), dtype=float)
    total = 0
    mean_acc = 0.
    assert(labels.shape[0] == predictions.shape[0])
    for i in range(labels.shape[0]):
        total += 1
        print('#%d: %s' % (total, i))
        pred = predictions[i]
        label = labels[i]
        flat_pred = np.ravel(pred)
        flat_label = np.ravel(label)
        
        acc = 0.
        for p, l in zip(flat_pred, flat_label):
            if l == 255:
                continue
            if l < nb_classes and p < nb_classes:
                conf_m[l, p] += 1
            else:
                print('Invalid entry encountered, skipping! Label: ', l,
                      ' Prediction: ', p, ' Img_num: ', img_num)

            if l==p:
                acc+=1
        acc /= flat_pred.shape[0]
        mean_acc += acc
    mean_acc /= total
    print 'mean acc: %f'%mean_acc
    
    I = np.diag(conf_m)
    U = np.sum(conf_m, axis=0) + np.sum(conf_m, axis=1) - I
    IOU = I/U
    meanIOU = np.mean(IOU)
    return conf_m, IOU, meanIOU


def evaluate(input_size, nb_classes):

    layer_per_block = [4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]
    model = Tiramisu(layer_per_block)

    model.load_weights('weights/prop_tiramisu_weights_67_12_func_10-e7_decay150.hdf5')

    test_data = np.load('./data/test_data.npy')
    test_label = np.load('./data/test_label.npy')
    assert(test_data.shape[0] == test_label.shape[0])

    test_pred = np.zeros(test_label.shape, dtype=float)
    for i in range(test_data.shape[0]):
        pred = model.predict(test_data[i:i+1])
        pred = np.argmax(np.squeeze(pred), axis=-1).astype(int)
        test_pred[i] = pred
    
    np.save("results/test_pred", test_pred)
    conf_m, IOU, meanIOU = calculate_iou(nb_classes, test_label, test_pred)
    print('IOU: ')
    print(IOU)
    print('meanIOU: %f' % meanIOU)
    print('pixel acc: %f' % (np.sum(np.diag(conf_m))/np.sum(conf_m)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dim', type=int, default=224)
    parser.add_argument('--classes', type=int, default=2)
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    evaluate((224, 224), args.classes)


