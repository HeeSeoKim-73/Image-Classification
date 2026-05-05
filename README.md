# Pokemon 이미지 분류 #

## 프로젝트 개요 ##
포켓몬 이미지를 입력받고, 해당 포켓몬의 이름을 분류하는 모델을 구현하였다.
2개의 모델(ResNet18, MobileNetV3)로 4개의 실험을 설정하여 성능을 비교하였다.

## 실험 종류 ##
- resnet18_freeze
- resnet18_finetune
- mobilenet_freeze
- mobilenet_finetune

## 실험 4개의 성능 비교 ##
모델 성능 평가는 Accuracy, Precision, Recall, F1-score을 지표로 이용하였다.
![성능 비교 결과](성능비교.png)

## Learning Curve ##
### ResNet18_Freeze ###
![resnet18_freeze_acc](resnet18_freeze_acc.png)
![resnet18_freeze_loss](resnet18_freeze_loss.png)
### ResNet18_Finetune ###
![resnet18_finetune_acc](resnet18_finetune_acc.png)
![resnet18_finetune_loss](resnet18_finetune_loss.png)
### MobileNet_Freeze ###
![mobilenet_freeze_acc](mobilenet_freeze_acc.png)
![mobilenet_freeze_loss](mobilenet_freeze_loss.png)
### MobileNet_Finetune ###
![mobilenet_finetune_acc](mobilenet_finetune_acc.png)
![mobilenet_finetune_loss](mobilenet_finetune_loss.png)
