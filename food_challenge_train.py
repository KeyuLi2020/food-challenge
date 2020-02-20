import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models

import os
import time
import shutil

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    begin = time.time()

    best_models_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            for data in dataloders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data.item()
                running_corrects += torch.sum(preds == labels).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_models_wts = model.state_dict()

    time_elapsed = time.time() - begin
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_models_wts)
    return model

if __name__ == '__main__':

    rawdata_dir = 'food_challenge2'
    rawdata_label_dir = 'food_challenge2/train.csv'

# os.makedirs('my-food-dataset')
# os.makedirs('my-food-dataset/train')
# os.makedirs('my-food-dataset/val')
#
# for i in range(4):
#     os.makedirs('my-food-dataset/train/%d' % (i))
#     os.makedirs('my-food-dataset/val/%d' % (i))

# for _, _, x in os.walk('food_challenge2/train'):
#     train_length = len(x)
#     print(train_length)

    label = np.array(pd.DataFrame(pd.read_csv(rawdata_label_dir), columns=['label'])).tolist()
    print(label)

    # for i in range(train_length):
    #     if i < train_length * 0.8:
    #         shutil.move('food_challenge2/train/%d.jpg' % (i), 'data/train/%d' % (label[i][0]))
    #     else:
    #         shutil.move('food_challenge2/train/%d.jpg' % (i), 'data/val/%d' % (label[i][0]))

    # 数据转换
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        }

    # 处理数据文件
    # 注意，用floyd上传dataset之后，需要：1 移除本地数据文件 2
    data_dir = '//my-food-dataset'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

    # wrap your data and label into Tensor
    # torchvision.datasets.ImageFolder只是返回list，list是不能作为模型输入的，
    # 因此在PyTorch中需要用另一个类来封装list，那就是：torch.utils.data.DataLoader。
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
              for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # 是否用GPU
    use_gpu = torch.cuda.is_available()

    # get model and replace the original fc layer with your fc layer
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 4)

    if use_gpu:
        model_ft = model_ft.cuda()

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model=model_ft,
                           criterion=criterion,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           num_epochs=25)
    # 保存模型
    torch.save(model_ft, 'net.pkl')





