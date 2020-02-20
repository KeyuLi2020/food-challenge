import torch
import os
import csv
import torch
from PIL import Image
from torchvision import transforms

device = torch.device('cuda')
transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def predict(img_path):

    result_list = []

    net = torch.load('net.pkl')
    net = net.to(device)
    torch.no_grad()

    for _, _, x in os.walk('/my-food-dataset/test'):
        predict_length = len(x)

    for each_img in range(predict_length):
        each_img_path = os.path.join('/my-food-dataset/test', '%d.jpg' % (each_img))
        img = Image.open(each_img_path)
        img = transform(img).unsqueeze(0)
        img_ = img.to(device)
        outputs = net(img_)

        _, predicted = torch.max(outputs, 1)
        # print(predicted)
        result_list.append([each_img, predicted[0].item()])

    return result_list


def write_result(filename, result):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for each_result in result:
            writer.writerow(each_result)

if __name__ == '__main__':
    model = torch.load('net.pkl')
    # print(model)
    result = predict('/my-food-dataset')
    write_result('predict_result.csv', result)

