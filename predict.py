import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn

import torchvision.models as models

from model import resnet34


def main(i):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path = i
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    #model = resnet34(num_classes=2).to(device)

    net = models.densenet121(pretrained=True)
    # print(net)
    # net.features.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    in_channel = net.classifier.in_features
    net.classifier = nn.Linear(in_channel, 2)
    weights_path = "./result_model_all/densenet121_rgb_p3_0.9192.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    net.load_state_dict(torch.load(weights_path, map_location=device))
    model = net
    model.to(device)
    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    #plt.show()


if __name__ == '__main__':
    import numpy as np
    val_images_path = np.load('.\\list\\test_images_path.npy')
    val_images_path = val_images_path.tolist()
    import os

    path_name = '../all/test/1H_del_black'  # 输入要获取文件的根目录
    for filename in os.listdir(path_name):
        print(filename)  # 输出获取的文件名
        i = path_name +'/' +filename
        main(i)
    """
    for i in val_images_path:
        print(i)
        main(i)
    """
