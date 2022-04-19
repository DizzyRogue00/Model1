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
from openpyxl import load_workbook
import pathlib
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
        terminal_stop=math.ceil(self.N/2)+1
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
                temp_dict={ii:m[i] for ii in temp.select(y,z)}
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

    def __Optimal(self,n,data=None,database=None,current_data=None):
        try:
            m=gp.Model('Bus_Collaborative')
            m.Params.timeLimit = 100
            m.setParam('nonconvex', 2)
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
            total_1=m.addVar(name='total_1')#with weight
            total_2=m.addVar(name='total_2')#without weight

            m.update()

            if n==1:
                index=[(x,y) for x in self.size for y in range(current_data[self.size.index(x)]+1)]
                index=gp.tuplelist(index)
                inter_tardy_1 = m.addVars(index,lb=-GRB.INFINITY, name='inter_tardy_1')
                inter_tardy_2 = m.addVars(index, name='inter_tardy_2')
                item1=gp.quicksum(self.lambda_[j-1]/2*pow(departure[j],2) for j in range(1,self.N+1))
                item2=gp.quicksum((in_vehicle[j+1]-in_vehicle_j.prod(self.p,'*',j+1))*tau[j+1] for j in range(1,self.N))
                item33=0
                item3=item33+gp.quicksum(w[j]*(self.t_bar(1)[j-1]-departure[j]) for j in range(1,self.N+1))
                item4=gp.quicksum(inter_tardy_2[j] for j in index)

                m.setObjective(self.theta[0]*item1+self.theta[1]*item2+self.theta[2]*item3+self.theta[3]*item4,sense=gp.GRB.MINIMIZE)

            else:
                index=[(x,y) for x in self.size for y in range(current_data[self.size.index(x)]-data[self.size.index(x)]+1)]
                index=gp.tuplelist(index)
                inter_tardy_1=m.addVars(index,lb=-GRB.INFINITY,name='inter_tardy_1')
                inter_tardy_2 = m.addVars(index, name='inter_tardy_2')
                item11=gp.quicksum(pow((departure[j]-database[data]['current_result'].getAttr('x',departure)[j]),2)*self.lambda_[j-1]/2 for j in range(1,self.N+1))
                item1=item11+gp.quicksum(self.lambda_[j-1]/2*pow((self.t_bar(n)[j-1]-departure[j]),2) for j in range(1,self.N+1))
                item2=gp.quicksum((in_vehicle[j+1]-in_vehicle_j.prod(self.p,'*',j+1))*tau[j+1] for j in range(1,self.N))
                item33=gp.quicksum(database[data]['current_result'].getAttr('x',w)[j]*(departure[j]-database[data]['current_result'].getAttr('x',departure)[j]) for j in range(1,self.N+1))
                item3=item33+gp.quicksum(w[j]*(self.t_bar(n)[j-1]-departure[j]) for j in range(1,self.N+1))
                item4=gp.quicksum(inter_tardy_2[j] for j in index)

                m.setObjective(self.theta[0]*item1+self.theta[1]*item2+self.theta[2]*item3+self.theta[3]*item4,sense=gp.GRB.MINIMIZE)

            if n==1:
                m.addConstr(in_vehicle_waiting==item2,name='in_vehicle_cost')
                m.addConstr(at_stop_waiting==item1,name='at_stop_cost')
                m.addConstr(extra_waiting==item33,name='extra_cost')
                m.addConstr(tardy_time == item4, name='tardiness_cost')
                m.addConstr(total_1==self.theta[0]*item1+self.theta[1]*item2+self.theta[2]*item33+self.theta[3]*item4,name='total_with_weight')
                m.addConstr(total_2==item1+item2+item33+item4, name='total_without_weight')
            elif n!= self.M:
                m.addConstr(in_vehicle_waiting==item2,name='in_vehicle_cost')
                m.addConstr(at_stop_waiting==item11,name='at_stop_cost')
                m.addConstr(extra_waiting==item33,name='extra_cost')
                m.addConstr(tardy_time == item4, name='tardiness_cost')
                m.addConstr(total_1==self.theta[0]*item11+self.theta[1]*item2+self.theta[2]*item33+self.theta[3]*item4,name='total_with_weight')
                m.addConstr(total_2==item11+item2+item33+item4, name='total_without_weight')
            else:
                m.addConstr(in_vehicle_waiting==item2,name='in_vehicle_cost')
                m.addConstr(at_stop_waiting==item1,name='at_stop_cost')
                m.addConstr(extra_waiting==item3,name='extra_cost')
                m.addConstr(tardy_time == item4, name='tardiness_cost')
                m.addConstr(total_1==self.theta[0]*item1+self.theta[1]*item2+self.theta[2]*item3+self.theta[3]*item4,name='total_with_weight')
                m.addConstr(total_2==item1+item2+item3+item4, name='total_without_weight')

            m.addConstr(departure[1]==self.headway*n,name='depart_1')
            m.addConstrs((departure[j]==departure[j-1]+self.ll[j-2]/self.v[j-2]+tau[j] for j in range(2,self.N+1)),name='depart')
            m.addConstrs((arrival[j]==departure[j-1]+self.ll[j-2]/self.v[j-2] for j in range(2,self.N+1)),name='arri')
            m.addConstrs((in_vehicle_j[j1,j2]==in_vehicle_j[j1,j2-1]*(1-self.p[j1,j2-1]) for j1,j2 in index_1 if j1!=j2-1),name='in_j')
            m.addConstrs((in_vehicle_j[j1,j2]==board[j1] for j1,j2 in index_1 if j1==j2-1),name='in_j_1')
            m.addConstrs((in_vehicle[j]==in_vehicle_j.sum('*',j) for j in range(2,self.N+2)),name='inTotal')

            if n==1:
                m.addConstrs((phi[j]==self.lambda_[j-1]*self.headway/2 for j in range(1,self.N+1)),name='waiting_1')
                m.addConstr(inter_tau_2==reduce(operator.add,map(lambda x,y:x*y,current_data[:-1],self.size))*self._unloading_rate/60,name='inter_tau_2_con')
                m.addConstrs((inter_tardy_1[k,q]==arrival[math.ceil(self.N/2)+1]-self.dd[k,1,0,q] for k,q in index), name='inter_tardy_1_con')
            else:
                m.addConstrs((phi[j]==self.lambda_[j-1]*(departure[j]-database[data]['current_result'].getAttr('x',departure)[j])+database[data]['current_result'].getAttr('x',w)[j] for j in range(1,self.N+1)),name='waiting')
                m.addConstrs((departure[j]-database[data]['current_result'].getAttr('x',departure)[j]>=0 for j in range(1,self.N+1)),name='overtaking_n_1')
                m.addConstrs((arrival[j]-database[data]['current_result'].getAttr('x',arrival)[j]>=0 for j in range(2,self.N+1)),name='overtaking_n_2')
                m.addConstr(inter_tau_2==reduce(operator.add,map(lambda x,y:x*y,map(lambda x,y:x-y,current_data[:-1],data[:-1]),self.size)) * self._unloading_rate / 60,name='inter_tau_2_con')
                m.addConstrs((inter_tardy_1[k,q]==arrival[math.ceil(self.N/2)+1]-self.dd[k,n,data[self.size.index(k)],q] for k,q in index),name='inter_tardy_1_con')
            m.addConstrs((alight[j]==in_vehicle.prod(self.p,'*',j) for j in range(2,self.N+2)),name='a1')
            m.addConstr(inter_board_limit_1==phi[1]-self.capacity,name='inter_board_limit_1_con')
            m.addConstrs((inter_board_limit[j]==phi[j]-(self.capacity-in_vehicle[j]+alight[j]) for j in range(2,self.N+1)),name='inter_board_limit_con')
            m.addConstr(w[1]==max_(0,inter_board_limit_1), name='moodify_w_1')
            m.addConstrs((w[j]==max_(0,inter_board_limit[j]) for j in range(2,self.N+1)),name='modify_w_')
            m.addConstrs((board[j]==phi[j]-w[j] for j in range(1,self.N+1)),name='factual_board')
            m.addConstrs((self.t_bar(n)[j-1]-departure[j]>=0 for j in range(1,self.N+1)),name='virtual')
            m.addConstrs((inter_tau_1[j] == board[j] * self.boarding_rate / 60 for j in range(2,self.N+1)), name='inter_tau_1_con')
            m.addConstrs((tau[j]==inter_tau_1[j] for j in range(2,self.N+1) if j!=math.ceil(self.N/2)+1),name='duration')
            m.addConstr(tau[math.ceil(self.N/2)+1]==max_(inter_tau_1[math.ceil(self.N/2)+1],inter_tau_2),name='duration_transship')
            m.addConstrs((inter_tardy_2[j]==max_(0,inter_tardy_1[j]) for j in index),name='inter_tardy_2_con')

            m.optimize()

            if m.status==GRB.OPTIMAL:
                print(m.status)
                self._m=m
                self._objVal = m.objVal
                self._result = m.getAttr('x', [in_vehicle_waiting, at_stop_waiting, extra_waiting, tardy_time,total_1,total_2])
                self._departure = m.getAttr('x', departure)
                self._arrival = m.getAttr('x', arrival)
                self._in_vehicle_j = m.getAttr('x', in_vehicle_j)
                self._in_vehicle = m.getAttr('x', in_vehicle)
                self._board = m.getAttr('x', board)
                self._w = m.getAttr('x', w)
                self._phi = m.getAttr('x', phi)
                self._tau = m.getAttr('x', tau)
                self._alight = m.getAttr('x', alight)
            elif m.status==GRB.TIME_LIMIT:
                m.Params.timeLimit = 200
                if m.MIPGap<=0.05:
                    print(m.status)
                    print(m.MIPGap)
                    self._m=m
                    self._objVal = m.objVal
                    self._result = m.getAttr('x', [in_vehicle_waiting, at_stop_waiting, extra_waiting, tardy_time, total_1,total_2])
                    self._departure = m.getAttr('x', departure)
                    self._arrival = m.getAttr('x', arrival)
                    self._in_vehicle_j = m.getAttr('x', in_vehicle_j)
                    self._in_vehicle = m.getAttr('x', in_vehicle)
                    self._board = m.getAttr('x', board)
                    self._w = m.getAttr('x', w)
                    self._phi = m.getAttr('x', phi)
                    self._tau = m.getAttr('x', tau)
                    self._alight = m.getAttr('x', alight)
                else:
                    m.Params.MIPGap = 0.05
                    m.optimize()
                    print("OK")
                    print(m.status)
                    self._m=m
                    self._objVal = m.objVal
                    self._result = m.getAttr('x', [in_vehicle_waiting, at_stop_waiting, extra_waiting,tardy_time,total_1,total_2])
                    self._departure = m.getAttr('x', departure)
                    self._arrival = m.getAttr('x', arrival)
                    self._in_vehicle_j = m.getAttr('x', in_vehicle_j)
                    self._in_vehicle = m.getAttr('x', in_vehicle)
                    self._board = m.getAttr('x', board)
                    self._w = m.getAttr('x', w)
                    self._phi = m.getAttr('x', phi)
                    self._tau = m.getAttr('x', tau)
                    self._alight = m.getAttr('x', alight)
            return self._m,self._objVal,self._result,self._departure, self._arrival, self._in_vehicle_j, self._in_vehicle, self._board, self._w, self._phi, self._tau, self._alight

        except gp.GurobiError as e:
            print('Error code'+str(e.errno)+': '+str(e))
        except AttributeError:
            print('Encounted an attribute error')

    def column_generation(self):
        self.demand_parcels()
        range_capacity=[self.max_disp.select(k,self.M) for k in self.size]
        range_capacity=[i for item in range_capacity for i in item]
        max_range_capacity=max(range_capacity)
        def compare(a,b):
            return reduce(operator.and_, map(lambda x, y: x <= y, a,b))
        result=list(filter(lambda x:compare(x,tuple(range_capacity)),product(range(max_range_capacity+1),repeat=len(self.size))))
        return result

    def dynamic_programming(self):
        column_name=self.column_generation()
        df=pd.DataFrame(columns=column_name,index=range(1,self.M+1))

        def cal_max(n):
            a=[self.max_disp.select(k,n) for k in self.size]
            b=[i for item in a for i in item]
            a=tuple(b)
            return a

        def compare(a,b):
            return reduce(operator.and_, map(lambda x, y: x <= y, a,b))

        database={}

        def cal_database_item(database,df,n,item):
            key=tuple(list(item)+[n])
            if n==1:
                if reduce(operator.add,map(operator.mul,item,self.size))<=self._parcel_capacity:
                    if compare(item,cal_max(n)):
                        self.__Optimal(n,current_data=key)
                        value={'previous':0,'current_result':self._m}
                        result_item={key:value}
                        df.loc[n][item]=self._result[4]
                        database.update(result_item)
            else:
                if compare(item,cal_max(n)):
                    for i in column_name:
                        if compare(i,item):
                            if reduce(operator.add,map(operator.mul,map(operator.sub,item,i),self.size))<=self._parcel_capacity:
                                previous_key=tuple(list(i)+[n-1])
                                if database[previous_key]['current_result'] is not inf:
                                    self.__Optimal(n,previous_key,database,key)
                                    if n==self.M:
                                        summation=df.loc[n-1][i]+self._objVal
                                    else:
                                        summation=df.loc[n-1][i]+self._result[4]
                                    if summation<=df.loc[n][item]:
                                        value={'previous':previous_key,'current_result':self._m}
                                        result_item={key:value}
                                        df.loc[n][item]=summation
                                        database.update(result_item)

        for n in range(1,self.M+1):
            df.loc[n]=inf
            for item in column_name:
                key = tuple(list(item) + [n])
                value = {'previous': 0, 'current_result': inf}
                result_item={key:value}
                database.update(result_item)
                #print(n)
                #print(item)
                #print(database)
                #print(df)
                cal_database_item(database,df,n,item)

        self.database=database
        self.df=df
        return self.database,self.df

    def __call__(self,*args,**kwargs):
        results=self.dynamic_programming()
        return results

    def directory(self,name:str):
        parent_dir=os.getcwd()
        filepath=os.path.join(parent_dir,str(name))
        if not os.path.exists(name):
            try:
                os.makedirs(filepath, exist_ok=True)
                print("Directory {} created successfully".format(name))
            except OSError as error:
                print("Directory {%s} cannot be created".format(name))
        else:
            print("Directory {} has existed".format(name))
        return filepath

    def combine_path(self,name:str,filename:str,fileformat:str):
        filepath=name+'/'+filename+'.'+fileformat
        return filepath

    def process(self):
        column_index=self.df.columns
        for i in range(len(column_index)-1,-1,-1):
            if self.df.loc[self.M][column_index[i]] is not inf:
                data=column_index[i]
                break
        final_data=tuple(list(data)+[self.M])

        def traverse(data,l=[]):
            if self.database[data]['previous'] == 0:
                l.insert(0,data)
                return l
            else:
                l.insert(0,data)
                return traverse(self.database[data]['previous'],l)

        def accumulate_diff(iterable,func):
            it=iter(iterable)
            try:
                total=next(it)
            except StopIteration:
                return
            yield total
            element1=total
            for element in it:
                total=func(element1,element)
                element1=element
                yield total

        def tuple_sub(a,b):
            return tuple(b[i]-a[i] for i in range(len(a)))

        result=traverse(final_data,[])
        result_slice=[i[:-1] for i in result]
        result_load=list(accumulate_diff(result_slice,tuple_sub))

        def tuple_prod(a,b):
            return reduce(operator.add,map(operator.mul,a,b))

        result_loadParcel=[tuple_prod(i,self.size) for i in result_load]

        return result,result_slice,result_load,result_loadParcel


    def Analysis(self):
        self.dynamic_programming()
        process_result=self.process()
        folder=self.directory('Results_ZC')

        def result_add(a,b):
            return [a[i]+b[i] for i in range(len(a))]

        result_list=[self.database[i]['current_result'].getAttr('x', [
            self.database[i]['current_result'].getVarByName('in_vehicle_waiting'),
            self.database[i]['current_result'].getVarByName('at_stop_waiting'),
            self.database[i]['current_result'].getVarByName('extra_waiting'),
            self.database[i]['current_result'].getVarByName('tardy_time'),
            self.database[i]['current_result'].getVarByName('total_1'),
            self.database[i]['current_result'].getVarByName('total_2')]) for i in process_result[0]]
        result_temp=reduce(result_add,result_list)
        result=result_temp[:4]+[result_temp[-1]]
        data_result = pd.DataFrame(columns=["In-vehicle", "At-stop", "Extra", "Tardiness","Total"], index=["FTNC"])
        data_result.loc['FTNC']=result
        filename='results_'+'Demand_'+str(self.demand)
        filedata_result_csv = self.combine_path(folder, filename, 'csv')
        data_result.to_csv(filedata_result_csv,mode='a+')
        filedata_result_excel = self.combine_path(folder, filename, 'xlsx')
        path=pathlib.Path(filedata_result_excel)
        sheet_name="FTNC_Demand"+str(self.demand)
        if path.exists():
            with pd.ExcelWriter(filedata_result_excel,engine='openpyxl',mode='a') as writer:
                data_result.to_excel(writer,sheet_name=sheet_name)
        else:
            with pd.ExcelWriter(filedata_result_excel,engine='openpyxl') as writer:
                data_result.to_excel(writer,sheet_name=sheet_name)

        def generate_in_vehicle(item:list):
            temp=[0]
            a=copy.deepcopy(item)
            temp.extend(a)
            return temp
        data_in_vehicle={str(i):
                             generate_in_vehicle(self.database[process_result[0][i-1]]['current_result'].getAttr('x',self.database[process_result[0][i-1]]['current_result'].getVarByName('in_vehicle'))) for i in range(1,self.M+1)}
        data_in_vehicle=pd.DataFrame(data_in_vehicle,index=range(1,self.N+2))
        #plt.figure(num=2, facecolor='white', edgecolor='black')
        markers_ZC=[".","^","1","s","*","+","x","D"]
        linestyle=['-','--','-.',':']*2
        color = ['#7B113A', "#150E56", "#1597BB", "#8FD6E1", "#E02401", "#F78812", "#Ab6D23", "#51050F"]
        style=list(map(lambda x,y:x+y,markers_ZC,linestyle))
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = 'Times New Roman'
        #plt.subplot(facecolor="w")
        #data_in_vehicle.plot(style=style, color=color, legend=True, linewidth=2.5)
        data_in_vehicle.plot(style=style, color=color, legend=True, linewidth=2.5)
        ax = plt.gca()
        ax.axhline(y=self.capacity, color='k', linestyle='-')
        ax.tick_params(top=False,bottom=True,left=True,right=False,direction='inout')
        ax.tick_params(which='major',width=1.5)
        ax.set_facecolor("w")
        ax.spines['bottom'].set(visible=True, color='k', linewidth=0.5)
        ax.spines['left'].set(visible=True, color='k', linewidth=0.5)
        plt.xticks(range(1, self.N + 2), fontweight='bold')
        plt.yticks(None, None, fontweight='bold')
        plt.xlabel("Bus Stations", fontdict=dict(fontweight='bold'))
        plt.ylabel('Load', fontweight='bold')
        legend_label=list(map(lambda x,y:x+y,['Bus ']*len(list(data_in_vehicle.columns)),list(data_in_vehicle.columns)))
        #ax.legend(loc='upper right')
        lg=ax.legend(legend_label,loc='upper left',bbox_to_anchor=(1.05,0.95),markerscale=1.5,framealpha=0.5,facecolor='white',edgecolor='black',borderaxespad=0)
        plt.grid(False)
        plt.tight_layout()
        #plt.show()
        filename="Passengers Loads under Freight Transport when demand is "+str(self.demand)
        filename_svg=self.combine_path(folder,filename,"svg")
        #plt.savefig('Results/Average passenger arrival rates at each bus stations.svg')
        plt.savefig(filename_svg,bbox_extra_artists=(lg,),bbox_inches='tight')
        filename_pdf = self.combine_path(folder, filename, "pdf")
        #plt.savefig('Results/Average passenger arrival rates at each bus stations.pdf',dpi=1000)
        plt.savefig(filename_pdf, dpi=1000,bbox_extra_artists=(lg,),bbox_inches='tight')

        #prepare data
        data_passenger=[
            [i,j,self.database[process_result[0][i-1]]['current_result'].getAttr('x',self.database[process_result[0][i-1]]['current_result'].getVarByName('board'))[j-1],
             self.database[process_result[0][i-1]]['current_result'].getAttr('x',self.database[process_result[0][i-1]]['current_result'].getVarByName('w'))[j-1],
             self.database[process_result[0][i-1]]['current_result'].getAttr('x',self.database[process_result[0][i-1]]['current_result'].getVarByName('phi'))[j - 1]
            ] for i in range(1,self.M+1) for j in range(1,self.N+1)
        ]
        data_passenger=pd.DataFrame(data_passenger,colimns=['Bus','Stop','Boarding','Still need to wait','Total wait'])
        x_var='Stop'
        groupby_var='Bus'
        data_b=data_passenger.groupby(groupby_var)
        bar_x=np.arange(1,self.N+1)
        bar_width=1/(self.M+1)
        bar_tick_label=list(map(lambda x:str(x),bar_x))
        colors = [plt.cm.Spectral(i / float(2 - 1)) for i in range(2)]
        #colors=['#8E354A','#261E47']
        plt.figure(num=3, facecolor='white', edgecolor='black')
        plt.rcParams['font.family']='serif'
        plt.rcParams['font.serif']='Times New Roman'
        for i,df in data_b:
            plt.bar(df[x_var]+bar_width*(i-1),df['Boarding'],width=bar_width,align='center',color=colors[0])
            plt.bar(df[x_var] + bar_width * (i - 1), df['Still need to wait'], bottom=df['Boarding'],width=bar_width, align='center',
                    color=colors[1])

        ax = plt.gca()
        ax.set_facecolor("w")
        ax.spines['bottom'].set(visible=True, color='k', linewidth=0.5)
        ax.spines['left'].set(visible=True, color='k', linewidth=0.5)
        ax.spines['top'].set(visible=True, color='k', linewidth=0.5)
        ax.spines['right'].set(visible=True, color='k', linewidth=0.5)
        plt.xticks(np.arange(1, self.N + 1) + bar_width * (self.M - 1) / 2, bar_tick_label,fontweight='bold')
        plt.yticks(None, None, fontweight='bold')
        plt.xlabel("Bus Stations", fontdict=dict(fontweight='bold'))
        plt.ylabel('Passengers', fontweight='bold')
        legend_label = ['Boarding','Stranded']
        # ax.legend(loc='upper right')
        lg = ax.legend(legend_label,loc='upper right',framealpha=0.5,
                       facecolor='white', edgecolor='black')
        plt.grid(False)
        filename="About to board and stranded passengers because of capacity limit under freight transport when demand is "+str(self.demand)
        filename_svg = self.combine_path(folder, filename, "svg")
        plt.savefig(filename_svg)
        filename_pdf = self.combine_path(folder, filename, "pdf")
        plt.savefig(filename_pdf,dpi=1000)

        x_var_average = 'Stop'
        data_b_average = data_passenger.groupby(x_var_average).mean()
        data_b_average=data_b_average[['Boarding', 'Still need to wait','Total wait']]
        #data_b_average=data_b_average['Boarding','Still need to wait','Total wait']
        bar_x = np.arange(1, self.N + 1)
        bar_width = 0.5
        bar_tick_label = list(map(lambda x: str(x), bar_x))
        #colors = [plt.cm.Spectral(i / float(2 - 1)) for i in range(2)]
        colors=['#9F353A','#66327C']
        #print(data_b_average)
        plt.figure(num=4, facecolor='white', edgecolor='black')
        plt.rcParams['font.family']='serif'
        plt.rcParams['font.serif']='Times New Roman'
        plt.bar(data_b_average.index,data_b_average['Boarding'],width=bar_width,align='center',color=colors[0])
        plt.bar(data_b_average.index,data_b_average['Still need to wait'],bottom=data_b_average['Boarding'],width=bar_width,color=colors[1],align='center')
        ax = plt.gca()
        ax.set_facecolor("w")
        ax.spines['bottom'].set(visible=True, color='k', linewidth=0.5)
        ax.spines['left'].set(visible=True, color='k', linewidth=0.5)
        ax.spines['top'].set(visible=True, color='k', linewidth=0.5)
        ax.spines['right'].set(visible=True, color='k', linewidth=0.5)
        plt.xticks(range(1,self.N+1),bar_tick_label,fontweight='bold')
        plt.yticks(fontweight='bold')
        plt.xlabel("Bus Stations", fontdict=dict(fontweight='bold'))
        plt.ylabel('Passengers', fontweight='bold')
        legend_label = ['Boarding', 'Stranded']
        ax.legend(legend_label, loc='upper right', framealpha=0.5,
                       facecolor='white')
        plt.grid(False)
        filename="Average number of about to board and stranded passengers because of capacity limit under freight transport when demand is "+str(self.demand)
        filename_svg = self.combine_path(folder, filename, "svg")
        plt.savefig(filename_svg)
        filename_pdf = self.combine_path(folder, filename,"pdf")
        plt.savefig(filename_pdf, dpi=1000)

        #route trajectory
        data_trajectory=[[self.database[process_result[0][i-1]]['current_result'].getAttr('x',self.database[process_result[0][i-1]]['current_result'].getVarByName('departure'))[0]] for i in range(1,self.M+1)]
        [
            data_trajectory[i-1].extend(
                [
                    self.database[process_result[0][i - 1]]['current_result'].getAttr('x', self.database[
                        process_result[0][i - 1]]['current_result'].getVarByName('arrival'))[j-1],
                    self.database[process_result[0][i - 1]]['current_result'].getAttr('x', self.database[
                        process_result[0][i - 1]]['current_result'].getVarByName('departure'))[j-1]
                ]
            )
            for i in range(1,self.M+1) for j in range(2,self.N+1)
        ]
        [
            data_trajectory[i-1].extend(
                [
                    self.database[process_result[0][i - 1]]['current_result'].getAttr('x', self.database[
                        process_result[0][i - 1]]['current_result'].getVarByName('departure'))[self.N-1]+self.ll[self.N-1]/self.v[self.N-1]
                ]
            )
            for i in range(1,self.M+1)
        ]
        data_trajectory=np.array(data_trajectory).transpose()
        column_name=list(map(str,range(1,self.M+1)))
        data_trajectory=pd.DataFrame(data_trajectory,columns=column_name)
        temp_list=list(zip(range(1,self.N+1),range(1,self.N+1)))
        temp_list=[list(i) for i in temp_list]
        temp_list=reduce(operator.add,temp_list)
        temp_list.pop(0)
        temp_list.append(self.N+1)
       #print(temp_list)
        data_trajectory['Stop']=temp_list
        print(data_trajectory)
        #print(type(data_trajectory))
        plt.figure(num=5, facecolor='white', edgecolor='black')
        plt.rcParams['font.family']='serif'
        plt.rcParams['font.serif']='Times New Roman'
        colors=[plt.cm.get_cmap('tab20b')(i/float(self.M-1)) for i in range(self.M)]
        color_count=0
        for col in data_trajectory.columns:
            if col !='Stop':
                plt.plot(data_trajectory[col],data_trajectory['Stop'],'.--',color=colors[color_count])
                color_count+=1

        ax=plt.gca()
        ax.set_facecolor('w')
        ax.spines['bottom'].set(visible=True, color='k', linewidth=0.5)
        ax.spines['left'].set(visible=True, color='k', linewidth=0.5)
        ax.spines['top'].set(visible=True, color='k', linewidth=0.5)
        ax.spines['right'].set(visible=True, color='k', linewidth=0.5)
        y_tick_label=list(map(str,range(1,self.N+1)))
        y_tick_label.append('1')
        plt.yticks(range(1,self.N+2),y_tick_label,fontweight='bold')
        plt.ylim(bottom=1)
        plt.xticks(None,None,fontweight='bold')
        plt.xlim(left=self.headway)
        ax.tick_params(top=False,bottom=True,left=True,right=False)
        ax.tick_params('y',which='major',direction='inout')
        ax.tick_params('x',which='both',direction='out')
        plt.xlabel('Time',fontdict=dict(fontweight='bold'))
        plt.ylabel('Bus Stations',fontweight='bold')
        legend_label=list(map(lambda x,y:x+y,['Bus ']*len(list(data_trajectory.columns[:-1])),list(data_trajectory.columns[:-1])))
        lg=ax.legend(legend_label,loc='upper left',bbox_to_anchor=(1,1),markerscale=1.5,framealpha=0.8,facecolor='w',borderaxespad=0)
        #lg=ax.legend(legend_label,loc='lower right')
        plt.tight_layout()
        #plt.show()
        filename="Bus Trajectory under Freight Transport when demand is "+str(self.demand)
        filename_svg = self.combine_path(folder, filename, "svg")
        plt.savefig(filename_svg,bbox_extra_artists=(lg,),bbox_inches='tight')
        filename_pdf = self.combine_path(folder, filename, "pdf")
        plt.savefig(filename_pdf, dpi=1000,bbox_extra_artists=(lg,),bbox_inches='tight')

    def Average_Analysis(self):
        repeat_time=5
        folder = self.directory('Results_ZC')

        def item_process():
            self.dynamic_programming()
            process_result=self.process()
            return self.database,self.df,process_result
        final_data_result=[item_process() for i in range(repeat_time)]
        data_temp_result = pd.DataFrame(columns=["In-vehicle", "At-stop", "Extra", "Tardiness", "Total"],
                                   index=range(1,repeat_time+1))

        def result_add(a,b):
            return [a[i]+b[i] for i in range(len(a))]

        def extract_data(item):
            database=item[0]
            data_result=item[2]
            result_list = [database[i]['current_result'].getAttr('x', [
                database[i]['current_result'].getVarByName('in_vehicle_waiting'),
                database[i]['current_result'].getVarByName('at_stop_waiting'),
                database[i]['current_result'].getVarByName('extra_waiting'),
                database[i]['current_result'].getVarByName('tardy_time'),
                database[i]['current_result'].getVarByName('total_1'),
                database[i]['current_result'].getVarByName('total_2')]) for i in data_result[0]]
            result_temp=reduce(result_add,result_list)
            result = result_temp[:4] + [result_temp[-1]]
            return result

        data_temp_result.loc[:]=[extract_data(i) for i in final_data_result]
        index_name="FTNC_Average_Demand_"+str(self.demand)
        data_result = pd.DataFrame(columns=["In-vehicle", "At-stop", "Extra", "Tardiness","Total"], index=[index_name])
        data_result.loc[index_name]=data_temp_result.mean()
        filename='Average_results_'+'Demand_'+str(self.demand)
        filedata_result_csv = self.combine_path(folder, filename, 'csv')
        data_result.to_csv(filedata_result_csv,mode='a+')
        filedata_result_excel = self.combine_path(folder, filename, 'xlsx')
        path=pathlib.Path(filedata_result_excel)
        sheet_name="FTNC_Average_Demand"+str(self.demand)
        if path.exists():
            with pd.ExcelWriter(filedata_result_excel,engine='openpyxl',mode='a') as writer:
                data_result.to_excel(writer,sheet_name=sheet_name)
        else:
            with pd.ExcelWriter(filedata_result_excel,engine='openpyxl') as writer:
                data_result.to_excel(writer,sheet_name=sheet_name)

        def extract_in_vehicle_data(item):
            database=item[0]
            data_result=item[2]
            result_list=[
                database[i]['current_result'].getAttr('x',database[i]['current_result'].getVarByName('in_vehicle'))
                for i in data_result[0]
            ]
            return result_list
        data_temp_in_vehicle_result = [extract_in_vehicle_data(i) for i in final_data_result]
        data_temp_in_vehicle=np.array(data_temp_in_vehicle_result)
        data_temp_in_vehicle_result=np.sum(data_temp_in_vehicle,0)
        data_temp_in_vehicle_df=pd.DataFrame(columns=range(2,self.N+2),index=range(1,self.M+1))
        data_temp_in_vehicle_df.loc[:]=data_temp_in_vehicle_result/repeat_time
        data_temp_in_vehicle_df.insert(0,1,0)
        data_in_vehicle=data_temp_in_vehicle_df.transpose()
        markers_ZC=[".","^","1","s","*","+","x","D"]
        linestyle=['-','--','-.',':']*2
        color = ['#7B113A', "#150E56", "#1597BB", "#8FD6E1", "#E02401", "#F78812", "#Ab6D23", "#51050F"]
        style=list(map(lambda x,y:x+y,markers_ZC,linestyle))
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = 'Times New Roman'
        #plt.subplot(facecolor="w")
        #data_in_vehicle.plot(style=style, color=color, legend=True, linewidth=2.5)
        data_in_vehicle.plot(style=style, color=color, legend=True, linewidth=2.5)
        ax = plt.gca()
        ax.axhline(y=self.capacity, color='k', linestyle='-')
        ax.tick_params(top=False,bottom=True,left=True,right=False,direction='inout')
        ax.tick_params(which='major',width=1.5)
        ax.set_facecolor("w")
        ax.spines['bottom'].set(visible=True, color='k', linewidth=0.5)
        ax.spines['left'].set(visible=True, color='k', linewidth=0.5)
        plt.xticks(range(1, self.N + 2), fontweight='bold')
        plt.yticks(None, None, fontweight='bold')
        plt.xlabel("Bus Stations", fontdict=dict(fontweight='bold'))
        plt.ylabel('Load', fontweight='bold')
        legend_label=list(map(lambda x,y:x+y,['Bus ']*len(list(data_in_vehicle.columns)),list(data_in_vehicle.columns)))
        #ax.legend(loc='upper right')
        lg=ax.legend(legend_label,loc='upper left',bbox_to_anchor=(1.05,0.95),markerscale=1.5,framealpha=0.5,facecolor='white',edgecolor='black',borderaxespad=0)
        plt.grid(False)
        plt.tight_layout()
        #plt.show()
        filename="Passengers Average Loads under Freight Transport when demand is "+str(self.demand)
        filename_svg=self.combine_path(folder,filename,"svg")
        #plt.savefig('Results/Average passenger arrival rates at each bus stations.svg')
        plt.savefig(filename_svg,bbox_extra_artists=(lg,),bbox_inches='tight')
        filename_pdf = self.combine_path(folder, filename, "pdf")
        #plt.savefig('Results/Average passenger arrival rates at each bus stations.pdf',dpi=1000)
        plt.savefig(filename_pdf, dpi=1000,bbox_extra_artists=(lg,),bbox_inches='tight')

        def extract_load_rate(item):
            data=item[2]
            return data

        data_parcel_load=[extract_load_rate(i) for i in final_data_result]/self._parcel_capacity/repeat_time
        data_parcel_load_rate=pd.DataFrame(data=data_parcel_load,index=range(1,self.M+1))
        colors = [plt.cm.Spectral(i / float(2 - 1)) for i in range(2)]
        plt.figure(num=3,facecolor='white',edgecolor='black')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = 'Times New Roman'
        plt.bar(data_parcel_load_rate.index,data_parcel_load_rate[0],width=0.6,color=colors[1],align='center')
        ax = plt.gca()
        ax.set_facecolor("w")
        ax.spines['bottom'].set(visible=True, color='k', linewidth=0.5)
        ax.spines['left'].set(visible=True, color='k', linewidth=0.5)
        ax.spines['top'].set(visible=True, color='k', linewidth=0.5)
        ax.spines['right'].set(visible=True, color='k', linewidth=0.5)
        plt.xticks(np.arange(1, self.M + 1),fontweight='bold')
        plt.yticks(None, None, fontweight='bold')
        plt.xlabel("Bus", fontdict=dict(fontweight='bold'))
        plt.ylabel('Goods Load Rate', fontweight='bold')
        plt.grid(False)
        filename = "Average Goods Load Rate under freight transport when demand is " + str(self.demand)
        filename_svg = self.combine_path(folder, filename, "svg")
        plt.savefig(filename_svg)
        filename_pdf = self.combine_path(folder, filename, "pdf")
        plt.savefig(filename_pdf,dpi=1000)

        #prepare data
        def extract_passengers_number(item):
            database=item[0]
            data=item[2]
            result_list1 = [database[i]['current_result'].getAttr('x',database[i]['current_result'].getVarByName('board')) for i in data[0]]
            result_list2 = [database[i]['current_result'].getAttr('x', database[i]['current_result'].getVarByName('w')) for i in data[0]]
            result_list3 = [database[i]['current_result'].getAttr('x', database[i]['current_result'].getVarByName('phi')) for i in data[0]]
            return result_list1,result_list2,result_list3

        data_board=[extract_passengers_number(i)[0] for i in final_data_result]
        data_w = [extract_passengers_number(i)[1] for i in final_data_result]
        data_phi = [extract_passengers_number(i)[2] for i in final_data_result]
        data_board=np.array(data_board)
        data_w=np.array(data_w)
        data_phi=np.array(data_phi)
        data_board_temp=np.sum(data_board,0)/repeat_time
        data_w_temp=np.sum(data_w,0)/repeat_time
        data_phi_temp=np.sum(data_phi,0)/repeat_time
        data_board_temp=data_board_temp.flatten()
        data_w_temp=data_w_temp.flatten()
        data_phi_temp=data_phi_temp.flatten()
        data_passenger=[
            [i,j,data_board_temp[(i-1)*self.N+j-1],data_w_temp[(i-1)*self.N+j-1],data_phi_temp[(i-1)*self.N+j-1]

            ] for i in range(1,self.M+1) for j in range(1,self.N+1)
        ]

        data_passenger=pd.DataFrame(data_passenger,colimns=['Bus','Stop','Boarding','Still need to wait','Total wait'])
        x_var='Stop'
        groupby_var='Bus'
        data_b=data_passenger.groupby(groupby_var)
        bar_x=np.arange(1,self.N+1)
        bar_width=1/(self.M+1)
        bar_tick_label=list(map(lambda x:str(x),bar_x))
        colors = [plt.cm.Spectral(i / float(2 - 1)) for i in range(2)]
        #colors=['#8E354A','#261E47']
        plt.figure(num=4, facecolor='white', edgecolor='black')
        plt.rcParams['font.family']='serif'
        plt.rcParams['font.serif']='Times New Roman'
        for i,df in data_b:
            plt.bar(df[x_var]+bar_width*(i-1),df['Boarding'],width=bar_width,align='center',color=colors[0])
            plt.bar(df[x_var] + bar_width * (i - 1), df['Still need to wait'], bottom=df['Boarding'],width=bar_width, align='center',
                    color=colors[1])

        ax = plt.gca()
        ax.set_facecolor("w")
        ax.spines['bottom'].set(visible=True, color='k', linewidth=0.5)
        ax.spines['left'].set(visible=True, color='k', linewidth=0.5)
        ax.spines['top'].set(visible=True, color='k', linewidth=0.5)
        ax.spines['right'].set(visible=True, color='k', linewidth=0.5)
        plt.xticks(np.arange(1, self.N + 1) + bar_width * (self.M - 1) / 2, bar_tick_label,fontweight='bold')
        plt.yticks(None, None, fontweight='bold')
        plt.xlabel("Bus Stations", fontdict=dict(fontweight='bold'))
        plt.ylabel('Passengers', fontweight='bold')
        legend_label = ['Boarding','Stranded']
        # ax.legend(loc='upper right')
        lg = ax.legend(legend_label,loc='upper right',framealpha=0.5,
                       facecolor='white', edgecolor='black')
        plt.grid(False)
        filename="About to board and stranded passengers because of capacity limit for several times under freight transport when demand is "+str(self.demand)
        filename_svg = self.combine_path(folder, filename, "svg")
        plt.savefig(filename_svg)
        filename_pdf = self.combine_path(folder, filename, "pdf")
        plt.savefig(filename_pdf,dpi=1000)

        x_var_average = 'Stop'
        data_b_average = data_passenger.groupby(x_var_average).mean()
        data_b_average=data_b_average[['Boarding', 'Still need to wait','Total wait']]
        #data_b_average=data_b_average['Boarding','Still need to wait','Total wait']
        bar_x = np.arange(1, self.N + 1)
        bar_width = 0.5
        bar_tick_label = list(map(lambda x: str(x), bar_x))
        #colors = [plt.cm.Spectral(i / float(2 - 1)) for i in range(2)]
        colors=['#9F353A','#66327C']
        #print(data_b_average)
        plt.figure(num=4, facecolor='white', edgecolor='black')
        plt.rcParams['font.family']='serif'
        plt.rcParams['font.serif']='Times New Roman'
        plt.bar(data_b_average.index,data_b_average['Boarding'],width=bar_width,align='center',color=colors[0])
        plt.bar(data_b_average.index,data_b_average['Still need to wait'],bottom=data_b_average['Boarding'],width=bar_width,color=colors[1],align='center')
        ax = plt.gca()
        ax.set_facecolor("w")
        ax.spines['bottom'].set(visible=True, color='k', linewidth=0.5)
        ax.spines['left'].set(visible=True, color='k', linewidth=0.5)
        ax.spines['top'].set(visible=True, color='k', linewidth=0.5)
        ax.spines['right'].set(visible=True, color='k', linewidth=0.5)
        plt.xticks(range(1,self.N+1),bar_tick_label,fontweight='bold')
        plt.yticks(fontweight='bold')
        plt.xlabel("Bus Stations", fontdict=dict(fontweight='bold'))
        plt.ylabel('Passengers', fontweight='bold')
        legend_label = ['Boarding', 'Stranded']
        ax.legend(legend_label, loc='upper right', framealpha=0.5,
                       facecolor='white')
        plt.grid(False)
        filename="Average number of about to board and stranded passengers because of capacity limit for several times under freight transport when demand is "+str(self.demand)
        filename_svg = self.combine_path(folder, filename, "svg")
        plt.savefig(filename_svg)
        filename_pdf = self.combine_path(folder, filename,"pdf")
        plt.savefig(filename_pdf, dpi=1000)













