import os
import sys
import json
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import metrics

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from my_dataset import MyDataSet
from utils import read_split_data, plot_data_loader_image

from model import resnet34,resnet50,resnet101
import torchvision.models as models

#root = "..\\ext_val"  # 数据集所在根目录

def main(i,train_images_path, train_images_label, val_images_path, val_images_label):
    i=i
    """"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    """
    #device= "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    #train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(root)

    data_transform = {
        "train": transforms.Compose([
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.RandomVerticalFlip(p=0.5),
                                     transforms.RandomAffine(degrees=(0, 180), scale=(0.9, 1.1)),
                                     transforms.Resize((256, 256)),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_data_set = MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])

    batch_size = 16

    #nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    nw = 4
    print('Using {} dataloader workers'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw,
                                               )

    val_data_set = MyDataSet(images_path=val_images_path,
                             images_class=val_images_label,
                             transform=data_transform["val"])
    val_num = len(val_data_set)
    validate_loader = torch.utils.data.DataLoader(val_data_set,
                                                  batch_size=2,
                                                  shuffle=False,
                                                  num_workers=nw)
    """
    net = models.densenet121(pretrained=True)
    #print(net)
    #net.features.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    in_channel = net.classifier.in_features
    net.classifier = nn.Linear(in_channel, 2)
    net.to(device)
    
   
    
    net = models.mobilenet_v2(pretrained=True)
    net.features._modules['0']._modules['0'] = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
    in_channel = net.classifier._modules['1'].in_features
    net.classifier._modules['1'] = nn.Linear(in_channel, 2)
    net.to(device)
    
    net = models.vgg11(pretrained=True)
    #print(net)
    #net.features._modules['0'] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    in_channel = net.classifier._modules['6'].in_features
    net.classifier._modules['6'] = nn.Linear(in_channel, 2)
    net.to(device)
    
    net = models.resnet101(pretrained=True)
    # print(net)
    #net.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 2)
    net.to(device)
    """
    net = models.vgg11(pretrained=True)
    # print(net)
    # net.features._modules['0'] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    in_channel = net.classifier._modules['6'].in_features
    net.classifier._modules['6'] = nn.Linear(in_channel, 2)
    net.to(device)


    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 60
    best_acc = 0.0

    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        y_true = []
        y_predict = []
        y_score = []
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                y_score_a_size = torch.softmax(outputs, dim=1)
                for l in range(len(y_score_a_size)):
                    y_score.append(y_score_a_size[l][1].cpu().detach().numpy().tolist())
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                y_true.append(val_labels.cpu().detach().numpy().tolist())
                y_predict.append(predict_y.cpu().detach().numpy().tolist())
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)
        y_true = np.array(y_true).reshape(-1)
        y_predict = np.array(y_predict).reshape(-1)
        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        #print('accuracy: {}'.format(accuracy_score(y_true, y_predict)))
        print("accuracy_score:", accuracy_score(y_true, y_predict))

        print("precision_score:", metrics.precision_score(y_true, y_predict))

        print("recall_score:", metrics.recall_score(y_true, y_predict))

        print("f1_score:", metrics.f1_score(y_true, y_predict))

        print("roc_auc_score:", metrics.roc_auc_score(y_true,y_score))
        b = metrics.roc_auc_score(y_true,y_score)
        save_path = './result_model_all/vgg11_rgb_p' + str(i) +'_'+str(round(b, 4))+ '.pth'
        if epoch >=39:
            if b > best_acc:
                beat_epoch =epoch+1
                best_acc = b
                torch.save(net.state_dict(), save_path)

    print('Finished Training,best epoch :',beat_epoch)


#root = "..\\all\\test"  # 数据集所在根目录
if __name__ == '__main__':
    """
    #torch.backends.cudnn.enabled = False
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(root)
    train_images_path = np.array(train_images_path)
    np.save('.\\list\\test_images_path.npy', train_images_path)

    train_images_label = np.array(train_images_label)
    np.save('.\\list\\test_images_label.npy', train_images_label)

    val_images_path = np.array(val_images_path)
    #np.save('val_images_path_add.npy', val_images_path)

    val_images_label = np.array(val_images_label)
    #np.save('val_images_label_add.npy', val_images_label)
    """


    #训练模型
    train_images_label = np.load('.\\list\\train_images_label.npy')
    train_images_label = train_images_label.tolist()

    train_images_path = np.load('.\\list\\train_images_path.npy')
    train_images_path = train_images_path.tolist()

    val_images_path = np.load('.\\list\\val_images_path.npy')
    val_images_path = val_images_path.tolist()

    val_images_label = np.load('.\\list\\val_images_label.npy')
    val_images_label = val_images_label.tolist()

    for i in range(0,5):
        main(i,train_images_path, train_images_label, val_images_path, val_images_label)
