from dataGenerator import train_valid_spliter,DataGenerator
from load_dict import load_dict

from model_conv_1D import conv_model_1d

dict1 = load_dict()
tvs = train_valid_spliter(dict1,3)

print('tvs generated')
train,valid = tvs.gen_train_valid_df()
print('train valid splited')

train_data = DataGenerator(TRAIN=True,subtype_dict=train,if_preprocess=True)
X,y = train_data[0]
print(X.shape,y.shape)

model = conv_model_1d()
model.predict(X,steps=1)
print('done')
#print(train_data.)