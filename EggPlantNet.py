
import os
import glob
import argparse
import numpy as np
from PIL import Image

import chainer
import chainer.links as L
from chainer import optimizers, training
from chainer import Chain
from chainer.training import extensions

NUM_CLASS = 4

# Download from http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel
caffemodel = './weight/VGG_ILSVRC_16_layers.caffemodel'
chainermodel = './weight/VGG_ILSVRC_16_layers.npz'

class EggPlantNet(Chain):
    def __init__(self, out_size, chainermodel=chainermodel):
        super(EggPlantNet, self).__init__(
            vgg = L.VGG16Layers(chainermodel),
            fc = L.Linear(None, out_size)
        )

    def __call__(self, x, train=True, extract_feature=False):
        with chainer.using_config('train', train):
            h = self.vgg(x, layers=['fc7'])['fc7']
            if extract_feature:
                return h
            y = self.fc(h)
        return y


def ParseArguments():
    # Argument Parser
    parse = argparse.ArgumentParser(description='EggPlant CNN')
    parse.add_argument('--load','-l', default='',
                       help='Load the Caffe Weights')
    parse.add_argument('--test', '-t', default='', help='This is a test')
    parse.add_argument('--pretrained_model', '-m',
                        help='Path to pretrained model file')
    parse.add_argument('--display_interval', type=int, default=1,
                        help='Interval of displaying log to console')
    parse.add_argument('--train_size', '-s', type=int, default=100)
    return parse.parse_args()


def LoadDataset():
    '''
    .
    ├── EggPlantNet.py
    └── images
        ├── 0
        |   ├── 001.jpg
        |   ├── 002.jpg
        |   └── ...
        ├── 1
        |   ├── 001.jpg
        |   ├── 002.jpg
        |   └── ...
        └── ...
    '''
    dataset = []
    current_dir = os.getcwd()
    imgs_dir = os.path.join(current_dir, 'images')
    dirs = os.listdir(imgs_dir)
    print(dirs)
    for dir in dirs:
        label = os.path.basename(dir)
        img_dir = os.path.join(imgs_dir, label)
        print(img_dir)
        print(os.path.join(img_dir, '*.jpg'))
        for f in glob.glob(os.path.join(img_dir, '*.jpg')):
            print(f)
            try:
                img = Image.open(os.path.join(dir,f))
            except:
                break

            img = L.model.vision.vgg.prepare(img)
            label = np.int32(label)
            dataset.append((img,label))

    return dataset



def main():
    args = ParseArguments()

    dataset = LoadDataset()
    dataset_size = len(dataset)
    print(dataset_size)

    model = L.Classifier(EggPlantNet(out_size=NUM_CLASS))
    alpha = 1e-4
    optimizer = optimizers.Adam(alpha=alpha)
    # optimizer = chainer.optimizers.MomentumSGD(0.05)
    optimizer.setup(model)
    # optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))
    model.predictor['fc'].W.update_rule.hyperparam.lr = alpha*10
    model.predictor['fc'].b.update_rule.hyperparam.lr = alpha*10

    gpu=0
    if gpu >= 0:
        chainer.cuda.get_device(gpu).use()
        model.to_gpu(gpu)

    # Freeze parameters
    # model.vgg.disable_update()

    epoch_num = 15
    validate_size = 30
    batch_size = 30

    # Shuffle Data & Divide into N size
    train, test = chainer.datasets.split_dataset_random(dataset, args.train_size)
    # set data & get iterator
    train_iter  = chainer.iterators.SerialIterator(train, batch_size)
    test_iter   = chainer.iterators.SerialIterator(test, batch_size, repeat=False, shuffle=False)
    # Update Setting
    updater = training.StandardUpdater(train_iter, optimizer, device=gpu)
    # Training Setting
    trainer = training.Trainer(updater, (epoch_num, 'epoch'), out='result')
    # Test Setting
    display_interval = (args.display_interval, 'epoch')
    trainer.extend(extensions.Evaluator(test_iter, model, device=gpu), trigger=display_interval)
    # Log Setting
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png'))
    print("Start Training")
    trainer.run()

    model.to_cpu()
    serializers.save_npz("mymodel.npz", model)

if __name__ == '__main__':
    main()
