from tokenize import group
from pre import coxDataLoader
import logging


import torch
from utils import utils
import os

from pycox.models import CoxTime
from pycox.models.cox_time import MLPVanillaCoxTime
from pycox.evaluation import EvalSurv
import torchtuples as tt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class COX:
    reader = 'coxDataLoader'

    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--model_path', type=str, default='',
                            help='Model save path.')
        parser.add_argument('--prediction_path', type=str, default='',
                            help='Model save path.')
        #in features
        return parser
    
 

    def __init__(self, args, corpus:coxDataLoader):
        self.corpus=corpus

        path_current=os.getcwd()
        path_s=path_current.split('/')
        baseDir=''
        for i in range(len(path_s)-1):
            baseDir+=path_s[i]
            baseDir+='/'
        model_path=baseDir+args.model_path
        self.model_path=model_path
        # print(self.model_path)
        self.prediction_path=args.prediction_path
        
        # self.define_model()
        return

    def define_model(self,args):
        num_nodes = [32, 32]
        batch_norm = True
        dropout = 0.1

        in_features=1 # click_rate, pctr, play_rate
        if args.play_rate==1:
            in_features+=1
        if args.pctr==1:
            in_features+=1
        in_features=self.corpus.x_train.shape[1]
        print("in_feature",in_features)
        net = MLPVanillaCoxTime(in_features, num_nodes, batch_norm, dropout)
        self.model = CoxTime(net, tt.optim.Adam, labtrans=self.corpus.labtrans)
    
    def load_model(self, model_path=None): #-> NoReturn:
        if model_path is None:
            model_path = self.model_path
        self.model.load_net(self.model_path)
        #self.load_state_dict(torch.load(model_path))
        logging.info('Load model from ' + model_path)

    def train(self):
        batch_size = 256
        # print("******",self.corpus.x_train.shape,self.corpus.y_train.shape)
        lrfinder = self.model.lr_finder(self.corpus.x_train, self.corpus.y_train, batch_size, tolerance=2)
        print("[train] best lr:",lrfinder.get_best_lr())
        self.model.optimizer.set_lr(0.01)
        epochs = 512
        callbacks = [tt.callbacks.EarlyStopping()]
        verbose = True
        log = self.model.fit(self.corpus.x_train, self.corpus.y_train, batch_size, epochs, callbacks, verbose,
                val_data=self.corpus.val.repeat(10).cat())
        _ = log.plot()
        # plt.savefig(".png")
        print("[train] model pll:",self.model.partial_log_likelihood(*self.corpus.val).mean())
        _ = self.model.compute_baseline_hazards()

        utils.check_dir(self.model_path)
        self.model.save_net(self.model_path)
        return
    
    def predict(self):
        # if os.path.exists(self.prediction_path+".csv"):
        #     logging.info("[prediction] exist prediction results: %s"%self.prediction_path)
        #     self.prediction_results=pd.read_csv(self.prediction_path+'.csv')
        #     return
        self.surv = self.model.predict_surv_df(self.corpus.x_test)
        # self.prediction_results=pd.DataFrame(columns=[])
        # print(self.surv)
        self.prediction_results=self.surv.T
        print("******",self.prediction_results.shape)
        self.corpus.df_test.reset_index(inplace=True)
        self.prediction_results['photo_id']=self.corpus.df_test['photo_id']
        utils.check_dir(self.prediction_path+'.csv')
        self.prediction_results.to_csv(self.prediction_path+'.csv')
        return
    
    def evaluate(self):
        self.surv.iloc[:, 10:20].plot()
        plt.ylabel('S(t | x)')
        _ = plt.xlabel('Time')
        plt.savefig(self.prediction_path+"_v0.png")
        plt.close()

        ev = EvalSurv(self.surv, self.corpus.durations_test, self.corpus.events_test, censor_surv='km')
        logging.info('ev.concordance_td: %f'%ev.concordance_td())
        print(ev.concordance_td())
        
        time_grid = np.linspace(self.corpus.durations_test.min(), self.corpus.durations_test.max(), 100)
        
        logging.info("integrated_brier_score: %f"%ev.integrated_brier_score(time_grid))
        print(ev.integrated_brier_score(time_grid))
        logging.info("ev.integrated_nbll: %f"%ev.integrated_nbll(time_grid))
        print(ev.integrated_nbll(time_grid))
        _ = ev.brier_score(time_grid).plot()
        plt.ylim(0,0.25)
        plt.xlim(0,300)
        plt.title("Brier Score")
        plt.savefig(self.prediction_path+"_v1.png")
        print("[evaluation] BS saved")
        # plt.show()

    def analysis(self,label,args):
        print(len(self.prediction_results))
        df=self.prediction_results
        print(df.columns)
        df=df.sample(frac=1)
        timeList=df.columns.tolist()
        df['SA_score']=0
        startTime=24
        endTime=24+12
        GROUP=10
        # print("******[analysis]******")
        # print(startTime,endTime,GROUP)
        logging.info("******[analysis]******")


        
        self.corpus.filtered_data()
        col_list=self.corpus.coxData.columns.tolist()
        baseline=self.corpus.coxData[['photo_id']]
        baseline['base_score']=0
        for i in col_list:
            if i!='photo_id':
                baseline['base_score']+=self.corpus.coxData[i]
        # baseline=baseline.sample(frac=1)
        # baseline['base_rank']=baseline['score'].rank(method='first',pct=True)
        # base_min=baseline['base_rank'].min()
        # baseline['base_rank']=baseline['base_rank']-base_min
        # baseline['base_group']=((baseline['base_rank'])*GROUP).astype('int')


        for i in range(startTime,endTime):
            if str(i) in timeList:
                print(str(i),df[str(i)].mean(),df[str(i)].std())
                df['SA_score']+=df[str(i)]

        print("[df], [baseline]",len(df),len(baseline))
        df=pd.merge(df,baseline,on='photo_id')
        print("[df]",len(df))

        df=df.sample(frac=1)
        df['base_rank']=df['base_score'].rank(method='first',pct=True)
        base_min=df['base_rank'].min()
        df['base_rank']=df['base_rank']-base_min
        df['base_group']=((df['base_rank'])*GROUP).astype('int')
        print(df['base_group'].value_counts())

        df=df.sample(frac=1)
        df['SA_rank']=df['SA_score'].rank(method='first',pct=True)
        SA_min=df['SA_rank'].min()
        df['SA_group']=((df['SA_rank']-SA_min)*GROUP).astype('int')
        print(df['SA_group'].value_counts())

        
        hourData=label.read_all(args)
        hourData['click']=hourData['click_rate']*hourData['counter']

        itemInfo=hourData[(hourData['timelevel']>=startTime)&(hourData['timelevel']<endTime)].groupby('photo_id'). \
            agg({'click':'sum','counter':'sum','play_rate':'mean','click_rate':'mean'})
        itemInfo['ctr']=itemInfo['click']/itemInfo['counter']

        # baseline=hourData[(hourData['timelevel']<startTime)].groupby('photo_id'). \
            # agg({'click':'sum','counter':'sum','play_rate':'mean','click_rate':'mean'})

        def getRank(g):
            return pd.DataFrame((1 + np.lexsort((g['play_rate'].rank(), \
            g['ctr'].rank())))/len(g), \
            index=g.index)
        # itemInfo['timelevel']=48
        itemInfo['per_rank']=itemInfo.groupby('timelevel').apply(getRank)
        per_min=itemInfo['per_rank'].min()
        itemInfo['per_rank']=itemInfo['per_rank']-per_min

        print(len(itemInfo),len(df))
        itemInfo=pd.merge(itemInfo,df,on='photo_id',suffixes=['_itemInfo',''],how='right') #,how='right'
        itemInfo.fillna(0,inplace=True)

        itemInfo['per_rank']=itemInfo.groupby('timelevel').apply(getRank)
        per_min=itemInfo['per_rank'].min()
        itemInfo['per_rank']=itemInfo['per_rank']-per_min

        print(len(itemInfo)) # (48-12 ~ 48+12) 10546; 55202 12229

        tagList=itemInfo['SA_group'].unique()
        tagList.sort()
        for tag in tagList:
            print("******")
            print(tag)
            tmp=itemInfo[itemInfo['SA_group']==tag]
            print(tmp['per_rank'].describe())
            x=tmp[tmp['per_rank']<0.1]
            print(len(x)/len(tmp),len(x),len(tmp))
        x=itemInfo['per_rank'].corr(itemInfo['SA_rank'],method='pearson')
        print(x)

        print("*********")
        tagList=itemInfo['base_group'].unique()
        tagList.sort()
        for tag in tagList:
            print("******")
            print(tag)
            tmp=itemInfo[itemInfo['base_group']==tag]
            print(tmp['per_rank'].describe())
            x=tmp[tmp['per_rank']<0.1]
            print(len(x)/len(tmp),len(x),len(tmp))
        x=itemInfo['per_rank'].corr(itemInfo['base_rank'],method='pearson')
        print(x)



