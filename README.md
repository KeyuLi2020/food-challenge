# food-challenge
I take part in a food classification competition. ResNet is used for classify.

Used "Floyd"( deep learning cloud computing platform)

Training Time: around 30mins

Dataset download: https://god.yanxishe.com/26

DataSet:
my-food-dataset
---val
    ---0
    ---1
    ---2
    ---3
---test
   ---0.jpg
   ...
   ---855.jpg
---train
    ---0
    ---1
    ---2
    ---3
    
Task:
Food Classification: 4 types of vegetables, 6000+ training dataset, 856 testing dataset
The Best Accuracy: 96.96+

Command Line:
floyd login -u likeyu2020

floyd data init likeyu2020/my-food-dataset

floyd data upload

floyd run --gpu --env pytorch-1.4 --data likeyu2020/datasets/my-food-dataset/2:my-food-dataset "python food_challenge_predict.py"
 


