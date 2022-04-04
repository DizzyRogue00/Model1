#from Model1 import *
import Model1 as mb
import Model1_holding as mh
import CollaborativeScenario as c
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
    zc=c.Collaborative(M=2,N=6)
    a,b,c,d=zc.demand_parcels()
    print(a)
    print(b)
    print(c)
    print(d)
    print(zc.dd)
