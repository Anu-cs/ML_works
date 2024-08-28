import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

dir = "D:\\ML_works\\archive\\train"
catagories= ['cat', 'dog']
data=[]

for catagory in catagories:
    path= os.path.join(dir, catagory)
    label= catagories.index(catagory)
    print(path)
    for img in os.listdir(path):
        img_path= os.path.join(path, img)
        img= cv2.imread(img_path)
        resize= cv2.resize(img, (50,50))
        flatten= resize.flatten()
        data. append([flatten,label])
print(len(data))

pick_in = open('data1.pickle', 'wb')
pickle.dump(data, pick_in)
pick_in.close()

pick_in = open('data1.pickle', 'rb')
data=pickle.load(pick_in)
pick_in.close()

random.shuffle(data)
features_train= []
labels_train= []
for feature, label in data:
    features_train.append(feature)
    labels_train.append(label)



dir = "D:\\ML_works\\archive\\val"
catagories= ['cat', 'dog']
data=[]

for catagory in catagories:
    path= os.path.join(dir, catagory)
    label= catagories.index(catagory)
    print(path)
    for img in os.listdir(path):
        img_path= os.path.join(path, img)
        img= cv2.imread(img_path)
        resize= cv2.resize(img, (50,50))
        flatten= resize.flatten()
        data. append([flatten,label])
print(len(data))

pick_in = open('data1.pickle', 'wb')
pickle.dump(data, pick_in)
pick_in.close()

pick_in = open('data1.pickle', 'rb')
data=pickle.load(pick_in)
pick_in.close()

random.shuffle(data)
features_test= []
labels_test= []
for feature, label in data:
    features_test.append(feature)
    labels_test.append(label)


model= SVC(C=1, kernel='poly', gamma='auto')
model.fit(features_train,labels_train)

pick=open('model.sav', 'wb')
pickle.dump(model, pick)
pick.close()
 



pick= open('model.sav','rb')
model1= pickle.load(pick)
pick.close()

prediction= model.predict(features_test)
accuracy=model.score(features_test, labels_test)

catagories= ['cat', 'dog']

print('Accuracy:', accuracy)
print('Prediction is:', catagories[prediction[0]])

mypet= features_test[0].reshape(50,50)
plt.imshow(mypet,cmap='gray')
plt.show()


