import numpy as np
import pandas as pd
import os 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



class data_loader():
    def __init__(self,path = "../Dataset/"):
        
        self.path = path

        self.genres=  list(os.walk(self.path))[0][1]
        
        self.scaler = MinMaxScaler()
        
        self.load_df()
        self.process_data()
        
        self.split_data()
        
    def load_df(self):

        df_dict={}
        for genre in self.genres[:]:
            df_list = []
            for plist in list(os.walk(self.path+genre))[0][2]:
                df_temp = pd.read_csv(self.path+genre+"/"+plist)
                df_list.append(df_temp)
            df_gen = pd.concat(df_list, axis=0).reset_index(drop=True)
            df_gen['genre'] = genre
            df_dict[genre]= df_gen
        self.df = pd.concat(df_dict.values(),axis=0).reset_index(drop=True)
        self.df.drop(['Unnamed: 0',"danceability.1"],axis=1,inplace=True)
        
    def process_data(self):
        #Binary columns
        df_bin   = self.df["explicit"]

        #Categorical columns
        df_cat   = self.df["time_signature"]
        #Converting categorical variables to dummy variables
        df_cat=pd.get_dummies(df_cat , drop_first=False,prefix= "time_sig")
        df_cat.rename(lambda x: "time_sig_"+str(x),axis="columns")


        #Numerical columns
        df_num   = self.df.drop(["explicit","name","album","artist","release_date","available_markets","genre"],axis=1)
 
        #Unrelated columns
        #df_trash = self.df[["name","album","artist","release_date","available_markets"]]
        
        #Concatting all data
        X_raw= pd.concat([df_num,df_cat,df_bin],axis=1).rename(str,axis="columns").reset_index(drop=True)
        X_raw['explicit']= X_raw['explicit'].apply(np.int16)
        X_raw= X_raw#[['popularity', 'danceability','acousticness','energy']]
        
        X = X_raw
        self.X = pd.DataFrame(self.scaler.fit_transform(X.to_numpy()),columns=X_raw.columns)
        
        #self.Y=pd.get_dummies(self.df["genre"] , drop_first=False,columns=self.df["genre"].unique())
        self.Y = pd.DataFrame(self.df['genre']).reset_index(drop=True)
        
    def split_data(self,test_size=0.3):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X,self.Y,test_size=0.3,shuffle=True)
        self.X_train, self.X_test, self.Y_train, self.Y_test =np.array(self.X_train), np.array(self.X_test), np.array(self.Y_train).reshape((self.Y_train.shape[0])), np.array(self.Y_test).reshape((self.Y_test.shape[0]))
        return self.X_train, self.X_test, self.Y_train, self.Y_test
    
#data_loader()