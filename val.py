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

from model import resnet34, resnet50, resnet101
import torchvision.models as models


# root = "..\\ext_val"  # 数据集所在根目录

def val(model, val_images_path, val_images_label):
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
    # device= "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(root)

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

    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
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
                                                  batch_size=1,
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
    net = model
    net.to(device)
    """
    net = models.mobilenet_v2(pretrained=True)

    print(net)
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    net.features._modules['0']._modules['0']= nn.Conv2d(1, 32, kernel_size=3, stride=2,padding=1, bias=False)
    in_channel = net.classifier._modules['1'].in_features
    net.classifier._modules['1'] = nn.Linear(in_channel, 2)
    net.to(device)
    """
    """
    net = resnet34()
    params = net.state_dict()
    for k, v in params.items():
        print(k)
    print(net)
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = "./resnet34-333f7ec4.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    net.conv1= nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 2)
    net.to(device)
    """
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

    y_true = np.array(y_true).reshape(-1)
    y_predict = np.array(y_predict).reshape(-1)
    val_accurate = acc / val_num
    # print('accuracy: {}'.format(accuracy_score(y_true, y_predict)))
    print("accuracy_score:", accuracy_score(y_true, y_predict))

    print("precision_score:", metrics.precision_score(y_true, y_predict))

    print("recall_score:", metrics.recall_score(y_true, y_predict))

    print("f1_score:", metrics.f1_score(y_true, y_predict))

    print("roc_auc_score:", metrics.roc_auc_score(y_true, y_score))
    b = metrics.roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    TP,FN,FP,TN =get(SR=y_predict,GT=y_true)
    print("TP:", float(torch.sum(TP)))
    print("FN:", float(torch.sum(FN)))
    print("FP:", float(torch.sum(FP)))
    print("TN:", float(torch.sum(TN)))

    #Acc = (TP + TN) / (TP + TN + FP + FN)
    CI = get_ci(n=float(torch.sum(TN)) + float(torch.sum(FN))+float(torch.sum(TP))+float(torch.sum(FP)),
                p=(float(torch.sum(TP)) + float(torch.sum(TN)))/ (float(torch.sum(TN)) + float(torch.sum(FN))+float(torch.sum(TP))+float(torch.sum(FP))))
    print("Acc CI:", CI)
    print("PPV,precision:", float(torch.sum(TP)) / (float(torch.sum(TP)) + float(torch.sum(FP))))
    CI = get_ci(n=float(torch.sum(TP)) + float(torch.sum(FP)),
                p=float(torch.sum(TP)) / (float(torch.sum(TP)) + float(torch.sum(FP))))
    print("PPV CI:",CI)

    print("NPV:", float(torch.sum(TN)) / (float(torch.sum(TN)) + float(torch.sum(FN))))
    CI = get_ci(n=float(torch.sum(TN)) + float(torch.sum(FN)),
                p=float(torch.sum(TN)) / (float(torch.sum(TN)) + float(torch.sum(FN))))
    print("NPV CI:", CI)

    print("SP:", float(torch.sum(TN)) / (float(torch.sum(TN)) + float(torch.sum(FP))))
    CI = get_ci(n=float(torch.sum(TN)) + float(torch.sum(FP)),
                p=float(torch.sum(TN)) / (float(torch.sum(TN)) + float(torch.sum(FP))))
    print("SP CI:", CI)
    print("SE,recall:", float(torch.sum(TP)) / (float(torch.sum(TP)) + float(torch.sum(FN))))
    CI = get_ci(n=float(torch.sum(TP)) + float(torch.sum(FN)),
                p=float(torch.sum(TP)) / (float(torch.sum(TP)) + float(torch.sum(FN))))
    print("SE CI:", CI)

    return fpr, tpr, thresholds,auc,y_score

def get_ci(n,p):
    from scipy import stats
    ci = stats.t.interval(0.95, n-1, p, np.sqrt(p * (1 - p)) / np.sqrt(n))
    return ci
def get(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SR = torch.tensor(np.array(SR))
    GT = torch.tensor(np.array(GT))

    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FN : False Negative
    # TP = ((SR==1)+(GT==1))==2
    # FN = ((SR==0)+(GT==1))==2
    TP = ((SR == 1) & (GT == 1))
    FN = ((SR == 0) & (GT == 1))
    FP = ((SR == 1) & (GT == 0))
    TN = ((SR == 0) & (GT == 0))

    # SE可能有问题，ｓｕｍ（ＴＰ＋ＦＮ）变为ｓｕｍ（ＴＰ）＋ｓｕｍ（ＦＮ）
    SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)
    SP =float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)
    return TP,FN,FP,TN

# root = "..\\all\\test"  # 数据集所在根目录
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

    # 训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_images_label = np.load('.\\list\\train_images_label.npy')
    train_images_label = train_images_label.tolist()

    train_images_path = np.load('.\\list\\train_images_path.npy')
    train_images_path = train_images_path.tolist()

    val_images_path = np.load('.\\list\\train_images_path.npy')
    val_images_path = val_images_path.tolist()

    val_images_label = np.load('.\\list\\train_images_label.npy')
    val_images_label = val_images_label.tolist()






    import matplotlib.pyplot as plt

    plt.figure()
    lw = 2

    print("----------resnet50-----------")
    net = models.resnet50(pretrained=True)
    # print(net)
    # net.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 2)
    weights_path = "./result_model_all/resnet50_rgb_p0_0.907.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    net.load_state_dict(torch.load(weights_path, map_location=device))

    fpr, tpr, thresholds, auc ,l_1= val(model=net, val_images_path=val_images_path, val_images_label=val_images_label)

    plt.plot(fpr, tpr, lw=lw, label='ResNet50')


    print("----------densenet121-----------")
    net = models.densenet121(pretrained=True)
    # print(net)
    # net.features.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    in_channel = net.classifier.in_features
    net.classifier = nn.Linear(in_channel, 2)
    weights_path = "./result_model_all/densenet121_rgb_p3_0.9192.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    net.load_state_dict(torch.load(weights_path, map_location=device))

    fpr, tpr, thresholds, auc ,l_2= val(model=net, val_images_path=val_images_path, val_images_label=val_images_label)

    plt.plot(fpr, tpr, lw=lw, label='DenseNet121')


    print("----------vgg11-----------")
    net = models.vgg11(pretrained=True)
    # print(net)
    # net.features._modules['0'] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    in_channel = net.classifier._modules['6'].in_features
    net.classifier._modules['6'] = nn.Linear(in_channel, 2)
    weights_path = "./result_model_all/vgg11_rgb_p3_0.8742.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    net.load_state_dict(torch.load(weights_path, map_location=device))

    fpr, tpr, thresholds, auc ,l_3= val(model=net, val_images_path=val_images_path, val_images_label=val_images_label)

    plt.plot(fpr, tpr, lw=lw, label='VGG11')

    print("----------mobilenet_v2-----------")
    net = models.mobilenet_v2(pretrained=True)
    net.features._modules['0']._modules['0'] = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
    in_channel = net.classifier._modules['1'].in_features
    net.classifier._modules['1'] = nn.Linear(in_channel, 2)
    weights_path = "./result_model_all/mobilenet_v2_rgb_p4_0.9036.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    net.load_state_dict(torch.load(weights_path, map_location=device))

    fpr, tpr, thresholds, auc ,l_4= val(model=net, val_images_path=val_images_path, val_images_label=val_images_label)

    plt.plot(fpr, tpr, lw=lw, label='Mobilenet_v2')



    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR(1-Specificity)')
    plt.ylabel('TPR(Sensitivity)')
    plt.title('Training Cohort')
    plt.legend(loc="lower right")
    plt.show()

    ###########################
    # 例如我们要存储两个list：name_list 和 err_list 到 Excel 两列
    # name_list和err_list均已存在
    #name_list = [10, 20, 30]  # 示例数据
    #err_list = [0.99, 0.98, 0.97]  # 示例数据

    # 导包，如未安装，先 pip install xlwt
    import xlwt

    # 设置Excel编码
    file = xlwt.Workbook('encoding = utf-8')

    # 创建sheet工作表
    sheet1 = file.add_sheet('sheet1', cell_overwrite_ok=True)

    # 先填标题
    # sheet1.write(a,b,c) 函数中参数a、b、c分别对应行数、列数、单元格内容
    sheet1.write(0, 0, "序号")  # 第1行第1列
    sheet1.write(0, 1, "label")  # 第1行第2列
    sheet1.write(0, 2, "resnet50")  # 第1行第3列
    sheet1.write(0, 3, "densenet121")  # 第1行第4列
    sheet1.write(0, 4, "vgg11")  # 第1行第5列
    sheet1.write(0, 5, "mobilenet_v2")  # 第1行第六列

    # 循环填入数据
    for i in range(len(val_images_label)):
        sheet1.write(i + 1, 0, i)  # 第1列序号
        sheet1.write(i + 1, 1, val_images_label[i])  # 第2列数量
        sheet1.write(i + 1, 2, l_1[i])  # 第3列误差
        sheet1.write(i + 1, 3, l_2[i])  # 第3列误差
        sheet1.write(i + 1, 4, l_3[i])  # 第3列误差
        sheet1.write(i + 1, 5, l_4[i])  # 第3列误差

    # 保存Excel到.py源文件同级目录
    file.save('Data_train.xls')

    ###########################
    #import main
    #main.DelongTest(l_2,l_1,val_images_label)
