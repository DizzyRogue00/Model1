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
#from operator import*
import operator
from numpy import inf
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
        temp = [(y, z) for y in range(1, self.N + 1) for z in range(y + 1, self.N + 2)]
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
                temp_dict={temp[y,z]:m[i]}
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
        due_left=sum(np.array(self.ll[:math.ceil(self.N/2)])/self.v[:math.ceil(self.N/2)])
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

    def __dispatch(self):
        parcels=random.randint(1,self._parcel_capacity)
        #print(parcels)
        size_reverse=self.size
        size_reverse.reverse()
        def div_mod(iterable,func=operator.mod,*,initial=None):
            it=iter(iterable)
            total=initial
            if initial is None:
                try:
                    total=next(it)
                except StopIteration:
                    return
            yield total
            div=0
            for element in it:
                div=total//element
                total=func(total,element)
                yield div
        dispatch_result=list(div_mod(size_reverse,initial=parcels))
        dispatch_result.pop(0)
        dispatch_result.reverse()
        #print(dispatch_result)
        return dispatch_result

    def __size_ready(self,data,k):#a piece of record
        size_k=self.size.index(k)
        ready_size_k=data[size_k]*[data[-2]]
        return ready_size_k

    def __size_due(self,data,k):#a piece of record
        size_k=self.size.index(k)
        due_size_k=data[size_k]*[data[-1]]
        return due_size_k

    def __max_disp(self,size_k,busNo,data):
        if data[size_k]!=[]:
            return reduce(operator.add,map(lambda x:1 if x<self.headway*busNo else 0,data[size_k]))
        else:
            return 0

    def dd_subscript_generate(self,t,data):
        if t==1:
            return [(i,t,0,q) for i in self.size for q in range(0,data[i,1]+1)]
        else:
            return [(i,t,p,q) for i in self.size for p in range(0,data[i,t-1]+1) for q in range(0,data[i,t]-p+1)]

    def dd_value_generate(self,data,due_Data):
        if data[3]==0:
            return 1000
        else:
            return due_Data[data[0]][data[2]+data[3]-1]

    def demand_parcels(self):
        ready_left,ready_right=self.release_interval()
        ready=list(map(lambda x: random.uniform(ready_left,ready_right), list(range(self.demand))))
        sorted_ready=sorted(ready)
        due_left,due_right=self.due_interval()
        max_due=0
        count_due=0
        sorted_due=[]
        while True:
            due_date=random.uniform(due_left,due_right)
            if due_date<=sorted_ready[count_due]:
                continue
            if due_date<max_due:
                continue
            else:
                sorted_due.append(due_date)
                count_due+=1
                max_due=due_date
                if count_due==len(sorted_ready):
                    break
        database={i+1:self.__dispatch()+[sorted_ready[i],sorted_due[i]] for i in range(len(sorted_ready))}
        database_ready={i: reduce(operator.add,map(lambda x:self.__size_ready(x,k=i),database.values())) for i in self.size}
        database_due={i: reduce(operator.add,map(lambda x: self.__size_due(x,k=i), database.values())) for i in self.size}
        max_disp_temp=gp.tuplelist([(i,j) for i in self.size for j in range(1,self.M+1)])
        max_disp={i:self.__max_disp(i[0],i[1],database_ready) for i in max_disp_temp}
        max_disp=gp.tupledict(max_disp)
        dd_temp=reduce(operator.add,map(lambda x:self.dd_subscript_generate(x,max_disp),range(1,self.M+1)))
        dd_temp=gp.tuplelist(dd_temp)
        dd={i:self.dd_value_generate(i,database_due) for i in dd_temp}
        dd=gp.tupledict(dd)
        self.database_ready=database_ready
        self.database_due=database_due
        self.max_disp=max_disp
        self.dd=dd
        return self.database_ready,self.database_due,self.max_disp,self.dd

    def __Optimal(self,n,data,database,current_data):
        try:
           m=gp.Model('Bus_Collaborative')
           index_1=gp.tuplelist([(y,z) for z in range(2,self.N+2) for y in range(1,z)])
           departure = m.addVars(range(1,self.N+1), name='departure')#departure time
           arrival=m.addVars(range(2,self.N+1),name='arrival')#arrival time
           in_vehicle_j=m.addVars(index_1,name='in_vehicle_j')#the number of ridership before arriving at the bus station from a specific bus station
           board=m.addVars(range(1,self.N+1),name='board')#boarding number
           in_vehicle=m.addVars(range(2,self.N+2),name='in_vehicle')#the number of ridership before arriving at the bus station
           phi=m.addVars(range(1,self.N+1),name='phi')#total waiting number
           alight=m.addVars(range(2,self.N+2),name='alight')#alighting number
           w=m.addVars(range(1,self.N+1),name='w')
           tau=m.addVars(range(2,self.N+1),name='tau')#dwelling time

           #
           inter_board_limit_1=m.addVar(lb=-GRB.INFINITY,name='inter_board_limit_1')
           inter_board_limit=m.addVars(range(2,self.N+1),lb=-GRB.INFINITY,name='inter_board_limit')
           inter_tau_1=m.addVars(range(2,self.N+1),name='inter_tau_1')
           inter_tau_2=m.addVar(name='inter_tau_2')
           #
           in_vehicle_waiting = m.addVar(name='in_vehicle_waiting')
           at_stop_waiting = m.addVar(name='at_stop_waiting')
           extra_waiting = m.addVar(name='extra_waiting')
           tardy_time=m.addVar(name='tardy_time')
           total=m.addVar(name='total')

           m.update()

           if n==1:

           else:
               index=[(x,y) for x in self.size for y in range(current_data[current_data.index(x)]-data[data.index(x)]+1)]
               index=gp.tuplelist(index)
               inter_tardy_1=m.addVars(index,name='inter_tardy_1')
               inter_tardy_2 = m.addVars(index, name='inter_tardy_2')
               item1=gp.quicksum(pow((departure[j]-database[data]['current_result'].getAttr('x',departure)[j]),2)*self.lambda_[j-1]/2 for j in range(1,self.N+1))
               item1+=gp.quicksum(self.lambda_[y-1]/2*pow((self.t_bar(n)[y-1]-departure[j]),2) for y in range(1,self.N+1))
               item2=gp.quicksum((in_vehicle[y+1]-in_vehicle_j.prod(self.p,'*',y+1))*tau[y+1] for y in range(1,self.N))
               item3=gp.quicksum(database[data]['current_result'].getAttr('x',w)[y]*(departure[j]-database[data]['current_result'].getAttr('x',departure)[j]) for j in range(1,self.N+1))
               item3+=1
               item4=gp.quicksum(inter_tardy_2[j] for j in index)
        except gp.GurobiError as e:
            print('Error code'+str(e.errno)+': '+str(e))
        except AttributeError:
            print('Encounted an attribute error')







