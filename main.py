#from Model1 import *
import Model1 as mb
import Model1_holding as mh
if __name__=="__main__":
    '''
    zc=mb.Model1(M=8,N=20)
    #a=zc.Optimal()
    #a=zc()
    #print(a)
    zc.Analysis()
    '''
    zc=mh.Model1(M=8,N=20)
    zc.Analysis()
