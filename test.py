import numpy as np
from tensorflow import keras
#import matplotlib.pyplot as plt

#读取文件
def read_data(filename):
    f=open(filename,'r')
    train_data=list()
    for line in f:
        train_data.append(line.strip())

    return train_data

#生成结构化数据集
def generate_dataset(data):
    train_data=list()
    train_target=list()
    for i in range(len(data)):
        if ',' in data[i]:
            domain=data[i].split(",")
            train_target.append(domain[-1])
            train_data.append(domain[0].replace('.',''))
        if ',' not in data[i]:
            train_data.append(data[i].replace('.',''))  
    if len(train_target)==0:
        return train_data 
    else:
        return train_data,train_target

#计算字符熵
def cal_entropy_letters(s):
    length=len(s)
    frequency=list()
    entropy=0
    for i in s:
        frequency.append(s.count(i)/length)
    for i in range(len(frequency)):
        entropy+=frequency[i]*np.log(frequency[i])
    return -1*entropy

#计算字符串中数字个数
def cal_num(s):
    num=0
    for i in range(len(s)):
        if s[i].isdigit():
            num+=1
    return num/len(s)

#特征选择
def generate_feature(data,target=None):
    train_data=np.zeros(shape=[len(data),3])
    train_target=np.zeros(shape=[len(data),1])
    #域名长度 字符熵 出现数字的频率
    for i in range(len(data)):
        train_data[i,0]=len(data[i])
        train_data[i,1]=cal_entropy_letters(data[i])
        train_data[i,2]=cal_num(data[i])
    #标注
    if(target!=None):
        for i in range(len(data)):
            if target[i].strip()=="dga":
                train_target[i,0]=1
            elif target[i].strip()=="notdga":
                train_target[i,0]=0
        assert(train_target.shape[0]==len(target))
        return train_data,train_target
    assert(train_data.shape[1]==3 and train_data.shape[0]==len(data))
    return train_data        

'''
#绘制学习曲线
def draw_curves(history):
    history_dict=history.history
    loss=history_dict['loss']
    val_loss=history_dict['val_loss']
    acc=history_dict['acc']
    val_acc=history_dict['acc']
    epochs=range(1,len(loss)+1)
    plt.plot(epochs, loss,color='r',label='Training loss') 
    plt.plot(epochs, val_loss, color='yellow',label='Validation loss') 
    plt.plot(epochs,acc,color='cyan',label='Training Accuracy')
    plt.plot(epochs,val_acc,color='g',label='Validation Accuracy')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss and Acc')
    plt.legend()
    plt.show()
'''

#训练神经网络
def build_model(data,target):
    model=keras.models.Sequential()
    model.add(keras.layers.Dense(6,activation='relu',input_dim=3))
    model.add(keras.layers.Dense(6,activation='relu'))
    model.add(keras.layers.Dense(1,activation='sigmoid'))
    model.compile(optimizer=keras.optimizers.RMSprop(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])
    model.fit(data,target,epochs=10,batch_size=8,validation_split=0.1)
    #history=model.fit(data,target,epochs=10,batch_size=8,validation_split=0.1)
    #draw_curves(history)
    return model

#预测测试集
def predict(model,test_data,test_str):
    test_target=model.predict_classes(test_data)
    for i in range(len(test_str)):
        if test_target[i]==1:
            test_str[i]+=",dga\n"
        else: 
            test_str[i]+=",notdga\n"
    with open("result.txt","w") as f:
        for i in range(len(test_str)):
            f.write(test_str[i])
    
    


if __name__ =="__main__":
    train_str=read_data("train.txt")
    test_str=read_data("test.txt")
    train_data,train_target=generate_dataset(train_str)
    test_data=generate_dataset(test_str)
    train_data,train_target=generate_feature(train_data,train_target)
    test_data=generate_feature(test_data)
    model=build_model(train_data,train_target)
    predict(model,test_data,test_str)
    



