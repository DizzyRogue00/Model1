import gurobipy as gp
import numpy as np
import scipy.stats as st
from gurobipy import *
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import os
import math
import copy
from functools import reduce
from itertools import *
import random
sb.set()

class Collaborative(object):
    def __init__(self,theta=[1,1,1,2],M=24,N=24):
        self.theta=theta
        self.M=M
        self.N=N
        self._parcel_capacity=30
        self._holding=3#mins
        self._unloading_rate=6# 6 sec per parcel

    @property
    def lambda_(self):
        #self._lambda_ = [0.6, 0.7, 0.75, 1, 0.8, 0.5, 1.2, 1.5, 1.3, 1.6, 2, 2, 3, 1.2, 2.2, 2, 1.2, 0.8, 1, 0.8,
         #                0.7, 0.6, 0.5, 0.5]
        self._lambda_ = [0.7, 0.75, 1, 0.8, 0.5, 1.2, 1.5, 1.3, 1.6, 2, 2, 3, 1.2, 2.2, 2, 1.2, 0.8, 1, 0.8,
                         0.7, 0.6, 0.5]
        return self._lambda_

    @lambda_.setter
    def lambda_(self, value):
        if not isinstance(value, list):
            raise ValueError('lambda must be a list')
        if len(value) != self.N:
            raise ValueError(f'The length must equal to {self.N}')
        self._lamnda_ = value

    @property
    def ll(self):
        self._ll = [800] * self.N
        return self._ll

    @ll.setter
    def ll(self, value):
        if not isinstance(value, list):
            raise ValueError('ll must be a list')
        if len(value) != self.N:
            raise ValueError(f'The length must equal to {self.N}')
        self._ll = value

    @property
    def v(self):
        self._v = [500] * self.N
        return self._v

    @v.setter
    def v(self, value):
        if not isinstance(value, list):
            raise ValueError('v must be a list')
        if len(value) != self.N:
            raise ValueError(f'The length must equal to {self.N}')
        self._v = value

    @property
    def p(self):
        def myiter(iterable, *, initial=None):
            it = iter(iterable)
            total = initial
            if initial == None:
                try:
                    total = next(it)
                except StopIteration:
                    return
            yield total
            alpha = 1
            for element in it:
                if element != 0:
                    element = element * alpha
                    total = element / (1 - total)
                    alpha *= total / element
                else:
                    total = 1
                yield total
        temp = [(x, y, z) for x in range(1, self.M + 1) for y in range(1, self.N + 1) for z in range(y + 1, self.N + 2)]
        temp = tuplelist(temp)
        if self.N%2==0:
            stop_step=self.N//2
        else:
            stop_step=(self.N+1)//2
        terminal_stop=self.N//2+1
        temp_={}
        for y in range(1,self.N+1):
            #m = np.array(list(np.arange(y + 1, self.N + 2)))
            m = np.array([0.5]*(self.N-y+1))
            if y>=terminal_stop:
                if len(m)==1:
                    m[-1]=1
                else:
                    m[-1]=0.5
                    m[:-1]=0.5/(self.N-y)
            else:
                m[:]=0.5/(stop_step-1)
                m[stop_step:]=0
                m[stop_step-y-1]=0.5
            m=list(myiter(m))
            i=0
            for z in range(y+1,self.N+2):
                temp_dict={ii:m[i] for ii in temp.select('*',y,z)}
                temp_.update(temp_dict)
                i+=1
        temp_=tupledict(temp_)
        self._p=temp_
        return self._p

    @p.setter
    def p(self, value):
        if not isinstance(value, tupledict):
            raise ValueError('p must be a tupledict')
        if len(value) != self.N * (self.N - 1) / 2:
            raise ValueError(f'The length must equal to {self.N * (self.N - 1) / 2}')
        self._p = value

    @property
    def headway(self):
        self._headway = 5
        return self._headway

    @headway.setter
    def headway(self, value):
        self._headway = value

    @property
    def boarding_rate(self):  # s/pax
        self._boarding_rate = 4
        return self._boarding_rate

    @boarding_rate.setter
    def boarding_rate(self, value):
        self._boarding_rate = value

    @property
    def capacity(self):
        self._capacity = 70
        return self._capacity

    @capacity.setter
    def capacity(self, value):
        self._capacity = value

    #demand will range from 5 to 20: [5,10,15,20]
    @property
    def demand(self):
        self._demand=5
        return self._demand

    @demand.setter
    def demand(self,value):
        self._demand=value
        return self._demand
    '''
    @property
    def t_bar(self):
        temp1 = np.array(self.ll)
        temp2 = np.array(self.v)
        temp3 = temp1 / temp2
        self._t_bar = [(self.M + 1) * self.headway + temp3[:j - 1].sum() + (j - 1) for j in range(1, self.N + 1)]
        return self._t_bar

    @t_bar.setter
    def t_bar(self, value):
        if not isinstance(value, list):
            raise ValueError('t_bar must be a list')
        if len(value) != self.N:
            raise ValueError(f'The length must equal to {self.N}')
        self._t_bar = value
        return self._t_bar
    '''
    @property
    def size(self):
        self._size=[1,2,5]
        return self._size

    @size.setter
    def size(self,value):
        if isinstance(value,list):
            raise ValueError('size should be a list')
        else:
            self._size=value
        return self._size

    def release_interval(self):
        return [0,self.headway*self.M]

    def due_interval(self):
        due_left=sum(np.array(self.ll[:math.ceil(self.N/2)])/self.v)
        due_right=self.headway*self.M+2*due_left
        return [due_left,due_right]

    def t_bar(self,value):
        temp1 = np.array(self.ll)
        temp2 = np.array(self.v)
        temp3 = temp1 / temp2
        intermediate_stop=math.ceil(self.N/2)+1
        duaration_buffer=[1 if i !=intermediate_stop else self._holding for i in range(2,self.N+1)]
        acc_time=list(map(lambda x,y:x+y,duaration_buffer,list(temp3)[:-1]))
        acc_time.insert(0,(value+1)*self.headway)
        self._t_bar=list(accumulate(acc_time))
        return self._t_bar
    def __demand_parcels(self):
        self.__parcelNo=[random.randint(1,self._parcel_capacity) for i in range(self.demand)]
        return self.__parcelNo
    def __generate_ready_date(self):
        ready_interval=self.release_interval()
        




