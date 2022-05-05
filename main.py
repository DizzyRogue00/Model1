#from Model1 import *
import Model1 as mb
import Model1_holding as mh
import CollaborativeScenario as c
#import CollaborativeScenario_holding as ch
import logging
if __name__=="__main__":
    '''
    zc=mb.Model1(M=8,N=20)
    #a=zc.Optimal()
    #a=zc()
    #print(a)
    zc.Analysis()
    '''
    '''
    zc=mh.Model1(M=8,N=20)
    zc.Analysis()    
    '''
    zc=c.Collaborative(M=3,N=3)
    #a,b,c,d=zc.demand_parcels()
    #print(d)
    #zc.Optimal(1, current_data=(0,0,0))
    #zc_database,zc_df=zc.dynamic_programming()
    #print(zc_database)
    #print(zc_df)
    #zc_df.to_csv('test_DP.csv')
    #zc.Analysis()
    zc.Average_Analysis()
'''
    no_control = mb.Model1(M=8, N=20)
    no_control.Analysis()
'''