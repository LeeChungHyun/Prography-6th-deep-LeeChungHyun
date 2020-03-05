# Prography-6th-deep-LeeChungHyun
TF2.0 keras 모델 클래스를 이용해 구현했다.

## Dataset
tf.keras.datasets.mnist

## train.py

```
class vgg_16(Model)
- __init__(self)
- call(self, inputs)
- model.compile()
- model.fit()
- model.save_weights()
```

## test.py

```
model.load_weights()
predict()
```

## Weight
https://drive.google.com/open?id=19Vxh9TJzVG1wyyjktxR99ofCj5b5Y_Vy <br>

## Accuracy
##### Training Accuracy #####
```
loss = [0.20133908291707436, 0.08781680787988007, 0.08026612276782592, 0.07344878148399293, 0.062208043841676164]
acc = [0.9465333, 0.9731333, 0.9752167, 0.97735, 0.98081666]
```

##### Testing Accuracy #####
```
acc = 97.5%
```

## Reference

1. https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer <br>

2. https://bskyvision.com/504 [VGG-16에대한 개념원리] <br>

3. https://www.tensorflow.org/guide/keras/overview?hl=ko <br>


