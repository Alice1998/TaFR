# -*- coding: UTF-8 -*-
import logging
import argparse
import pandas as pd
from pre.coxDataLoader import *
import numpy as np
import os

class Label:
    def parse_data_args(parser):
        parser.add_argument('--label_path', type=str, default='',
                            help='died info csv file')
        parser.add_argument('--agg_hour', type=int, default=1,
                            help='group hour')
        parser.add_argument('--end_time', type=int, default=24*7,
                            help='group hour')
        parser.add_argument('--EXP_hazard', type=float, default=0.5,
                            help='group hour')
        parser.add_argument('--noEXP_hazard', type=float, default=0.5,
                            help='group hour')
        parser.add_argument('--acc_thres', type=float, default=-3,
                            help='group hour')
        parser.add_argument('--prepareLabel', type=int, default=0,
                            help='prepareDied')
        parser.add_argument('--exposed_duration',type=int,default=0)
        return parser

      
    def __init__(self,args,corpus): #dataLoader
        self.agg_hour=args.agg_hour
        self.end_time=args.end_time
        self.EXP_hazard=args.EXP_hazard
        self.noEXP_hazard=args.noEXP_hazard
        self.acc_thres=args.acc_thres
        self.prepareLabel=args.prepareLabel
        self.exposed_duration=args.exposed_duration
        if args.label_path=='':
            log_args = [args.dataset,\
                str(self.agg_hour),str(corpus.start_time),str(self.end_time),\
                str(self.EXP_hazard),str(self.noEXP_hazard),str(self.acc_thres)]
            log_file_name = '__'.join(log_args).replace(' ', '__')
            if args.label_path == '':
                if args.exposed_duration:
                    args.label_path = '../label/{}_v2.csv'.format(log_file_name)
                else:
                    args.label_path = '../label/{}.csv'.format(log_file_name)
        self.label_path=args.label_path
        print("label path",self.label_path)
        if not os.path.exists(self.label_path) or self.prepareLabel:
            self.prepareDied(args,corpus)
        # else:
        #     self.prepareDied(corpus)

        return
    
    def read_all(self,args):
        dataFolder=args.path+args.dataset
        df=pd.DataFrame()
        if args.dataset=='kwai':
            df=pd.read_csv('%s/%s_10F_167H_hourLog.csv'%(dataFolder,args.dataset[5:]))
            df.rename({"expose_hour":'timelevel','is_click':'click_rate','new_pctr':'pctr'},axis=1,inplace=True)
        elif args.dataset=='MIND':
            df=pd.read_csv('%s/itemHourLog.csv'%(dataFolder))
            df.rename({"item_id":'photo_id','is_click':'click_rate','new_pctr':'pctr'},axis=1,inplace=True)
        else:
            for i in range(10):
                if args.exposed_duration:
                    tmp=pd.read_csv("%s/%s_%dsystem_itemHour_log.csv"%(dataFolder,args.dataset[5:],i))
                else:
                    tmp=pd.read_csv("%s/%s_%dsystem_itemHour_log.csv"%(dataFolder,args.dataset[5:],i))
                print('[loader]',i)
                # print(tmp.columns.tolist())
                df=pd.concat([df,tmp],ignore_index=True)
        # df.reset_index(inplace=True)
        print(df.columns.tolist())
        if self.exposed_duration:
            df.rename({'timelevel':'timelevel_old','timelevel_exposed':'timelevel'},axis=1,inplace=True)
        print(df.columns.tolist())
        df=df[df['timelevel']<self.end_time].copy()
        if 'play_time' in df.columns.tolist():
            df['play_rate']=df['play_time']/df['photo_time']
        return df

    def prepareDied(self,args,corpus):
        # self.hourData=self.read_all(args)
        # self.coxData=pd.read_csv("%s/cox.csv"%self.dataFolder)
        # self.hourData=self.read_all(args)
        hourInfo=self.read_all(args) #corpus.hourData
        print(corpus.play_rate,corpus.pctr)
        # click_rate, play_rate, pctr
        second_label='play_rate'
        if second_label not in hourInfo.columns.tolist():
            second_label='exposure'
        def getRank(g):
            return pd.DataFrame((1 + np.lexsort((g[second_label].rank(), \
            g['click_rate'].rank())))/len(g), \
            index=g.index)
        # print(hourInfo.columns)
        hourInfo=hourInfo.sample(frac=1)
        hourInfo['riskRank']=hourInfo.groupby(['timelevel']).apply(getRank)
        print(hourInfo['riskRank'].describe())

        def getFlag(v,t):
            if t<corpus.start_time:
                return 0
            return v-self.EXP_hazard+self.noEXP_hazard #min(v-self.EXP_hazard,0) #
        
        hourInfo['riskFlag']=hourInfo.apply(lambda v:getFlag(v['riskRank'],v['timelevel']),axis=1)
        logging.info(hourInfo['riskFlag'].describe())
        # hourInfo.sort_values(['photo_id','timelevel'],inplace=True)

        #type=0
        index_list=[]
        ids_list=[]
        time_list=[]
        for id in hourInfo['photo_id'].unique():
            for time in range(corpus.start_time,self.end_time):
                index_list.append("%d-%d"%(id,time))
                ids_list.append(id)
                time_list.append(time)
        new_system=pd.DataFrame({'photo_id':ids_list,'timelevel':time_list,'tag':index_list})
        hourInfo['tag']=hourInfo['photo_id'].astype('str')+'-'+hourInfo['timelevel'].astype('str')
        new_system=pd.merge(new_system,hourInfo[['tag','riskFlag']],on='tag',how='left')
        new_system.fillna(0,inplace=True)
        new_system.sort_values(['photo_id','timelevel'],inplace=True)
        new_system['risk']=new_system.groupby('photo_id')['riskFlag'].cumsum()
        new_system['risk']=new_system['risk']-(new_system['timelevel']-corpus.start_time)*self.noEXP_hazard

        # new_system.to_csv(self.label_path,index=False)
        # print("finish labeling")
        # return

        died=new_system[(new_system['risk']<=self.acc_thres)&(new_system['timelevel']>=corpus.start_time)].copy()
        died['died']=1
        died.sort_values('timelevel',inplace=True)
        died=died.groupby('photo_id').head(1)

        remain=new_system[~new_system['photo_id'].isin(died['photo_id'].tolist())].copy()
        remain['died']=0
        remain.sort_values('timelevel',inplace=True)
        remain=remain.groupby('photo_id').tail(1)

        item=pd.concat([remain,died])
        item.to_csv(self.label_path,index=False)
        logging.info(item.describe())

        return