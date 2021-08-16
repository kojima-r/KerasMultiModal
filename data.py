import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.utils import load_img
from sklearn.model_selection import train_test_split

from sklearn.utils import class_weight

class SampleDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df_img, df_microarray, batch_size, num_classes,shuffle=True):
        df_ex=df_img[['FINDING_TYPE_x','BARCODE',"PATH"]]
        img_data=[]
        valid_barcode=set()
        label_set=set()
        ys=[]
        for index, row in df_ex.iterrows():
            barcode =row["BARCODE"]
            path=row["PATH"]
            y   =row["FINDING_TYPE_x"]
            valid_barcode.add(barcode)
            label_set.add(y)
            img_data.append((path,barcode,y))
        microarray_data={}
        for index, row in df_microarray.iterrows():
            key =row["BARCODE"]
            if key in valid_barcode:
                data=row.iloc[1:].values
                microarray_data[key]=data

        self.label_list=list(label_set)
        
        ys=[]
        for el in img_data:
            path,barcode,y = el
            ys.append(self.label_list.index(y))         
        c_weight = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(ys),  y=ys)
        
        #print(self.label_list)
        #print(c_weight)

        self.class_weight={c:w for c,w in enumerate(c_weight)}
        self.microarray_data=microarray_data
        self.img_data=img_data
        self.batch_size=batch_size
        self.num_classes = num_classes
        self.num_samples=len(img_data)
        self.idx=np.arange(self.num_samples)
        self.shuffle=shuffle
        self.iter=0

    def __len__(self):
        return self.num_samples//self.batch_size
    
    def __getitem__(self,index):
        batch_index = self.idx[index*self.batch_size:(index+1)*self.batch_size]
        return self.__get_data(batch_index)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idx)
        
    def __get_data(self,batch):
        X1=[]
        X2=[]
        Y=[]
        for i in batch:
            path,barcode,y = self.img_data[i]
            x1_pil=load_img(path)
            x1=np.asarray(x1_pil) 
            x2=self.microarray_data[barcode]
            y_i=self.label_list.index(y)
            yy=np.zeros((self.num_classes,))
            yy[y_i]=1
            X1.append(x1)
            X2.append(x2)
            Y.append(yy)
        return ({"img":np.array(X1,dtype=np.float32),"microarray":np.array(X2,dtype=np.float32)}, np.array(Y,dtype=np.int32))
    def gen_func(self):
        while(True):
           yield self.__getitem__(self.iter)
           self.iter+=1
           print("*",self.iter)
           if(self.iter>=self.__len__()):
               self.iter=0
def make_dataset(g, strategy=None):
    
    strategy =None
    ds = tf.data.Dataset.from_generator((lambda: g.gen_func()),
            output_types=  ({'img': tf.float32, 'microarray': tf.float32}, tf.int32),
            output_shapes= ({'img': (None,512,512,3), 'microarray': (None,4835)}, (None,2)))
    
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA#.OFF
    ds = ds.with_options(options)

    # Optional: Make it a distributed dataset if you're using a strategy
    if strategy is not None:
        ds = strategy.experimental_distribute_dataset(ds)
    
    return ds
     
def main():
    df = pd.read_csv('small_train.csv', index_col=0)
    df_microarray = pd.read_csv('repeatliver_myegene_dropna_normalized.csv',header=0)

    df = df.sample(frac=1).reset_index(drop=True)
    df_train, df_val = train_test_split(df, stratify=df['FINDING_TYPE_x'])

    g=SampleDataGenerator(df_train, df_microarray, batch_size=3, num_classes=2,shuffle=False)
    x=g.__getitem__(0)
    #print(x)
    print(x[0]["img"].shape,x[0]["microarray"].shape,x[1].shape)
    ###
    ds = make_dataset(g)

    #(tf.float32,tf.float32, tf.int32))#, ((tf.TensorShape([None,512,512,3]),tf.TensorShape([None,None])), tf.TensorShape([None])) ) 
    for value in ds.take(1):
        print(value)

if __name__ == "__main__":
    main()
