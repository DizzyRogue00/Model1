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
sb.set()

class Model1(object):
    def __init__(self, theta=[1, 1, 2], M=24, N=24, t_0=0):
        self.theta = theta
        self.M = M
        self.N = N
        self.t_0 = t_0

    @property
    def e_i(self):
        self._e_i = [0] * self.M
        return self._e_i

    @e_i.setter
    def e_i(self, value):
        if not isinstance(value, list):
            raise ValueError('ei must be a list')
        else:
            if len(value) != self.M:
                raise ValueError(f'The length must equal to {self.M}')
            else:
                self._e_i = value

    @property
    def lambda_(self):
        #self._lambda_ = [0.6, 0.7, 0.75, 1, 0.8, 0.5, 1.2, 1.5, 1.3, 1.6, 2, 2, 3, 1.2, 2.2, 2, 1.2, 0.8, 1, 0.8,
        #                 0.7, 0.6, 0.5, 0.5]
        self._lambda_ = [ 0.7, 0.75, 1, 0.8, 0.5, 1.2, 1.5, 1.3, 1.6, 2, 2, 3, 1.2, 2.2, 2, 1.2, 0.8, 1, 0.8,
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
    def l(self):
        self._l = [0] * self.M
        return self._l

    @l.setter
    def l(self, value):
        if not isinstance(value, list):
            raise ValueError('l must be a list')
        if len(value) != self.M:
            raise ValueError(f'The length must equal to {self.M}')
        self._l = value

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
        '''
        loc = 0
        temp_ = {}
        for y in range(1, self.N + 1):
            m = np.array(list(range(y + 1, self.N + 2))) / 10
            m = st.lognorm.pdf(m, 1, loc=loc, scale=1)
            m = m / m.sum()
            i = 0
            for z in range(y + 1, self.N + 2):
                temp_dict = {ii: m[i] / m[i:].sum() for ii in temp.select('*', y, z)}
                temp_.update(temp_dict)
                i += 1
            loc += 0.1
        temp_ = tupledict(temp_)
        self._p = temp_
        return self._p
        '''
        '''
        loc=self.N//2+1
        scaler=(self.N-loc)//3+1
        temp_={}
        for y in range(1,self.N+1):
            m=np.array(list(range(y+1,self.N+2)))
            m=st.norm.pdf(m,loc=loc,scale=scaler)
            m=m/m.sum()
            i=0
            for z in range(y+1,self.N+2):
                temp_dict={ii:m[i]/m[i:].sum() for ii in temp.select('*',y,z)}
                temp_.update(temp_dict)
                i+=1
        temp_=tupledict(temp_)
        self._p=temp_
        return self._p
        '''

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

    def __Optimal(self):
        try:
            m = gp.Model('bus_original')
            m.Params.timeLimit = 100
            m.setParam('nonconvex', 2)
            # departure1=m.addVar(lb=0.0,ub=GRB.INFINITY,vtype=GRB.CONTINUOUS,name='departure1')
            # index_1=gp.tuplelist([x,y] for x in range(1,self.M+1) for y in range(self._e_i[x-1]+1,self.N+1))
            # index_arrival=gp.tuplelist([x,y] for x in range(1,self.M+1) for y in range(self._e_i[x-1]+1,self.N+1))
            # index_2 = gp.tuplelist([(x, y, z) for x in range(1, self.M + 1) for z in range(self._e_i[x - 1] + 2, self.N + 2) for y in range(1, z)])
            index_1 = gp.tuplelist([(x, y) for x in range(1, self.M + 1) for y in range(1, self.N + 1)])
            index_2 = gp.tuplelist([(x, y) for x in range(1, self.M + 1) for y in range(2, self.N + 1)])
            index_3 = gp.tuplelist(
                [(x, y, z) for x in range(1, self.M + 1) for z in range(2, self.N + 2) for y in range(1, z)])
            index_4 = gp.tuplelist([(x, y) for x in range(1, self.M + 1) for y in range(2, self.N + 2)])
            # index_5 = gp.tuplelist([(x, y) for x in range(1, self.M + 1) for y in range(2, self.N + 2)])
            # index_6 = gp.tuplelist([(x, y) for x in range(1, self.M + 1) for y in range(2, self.N + 1)])

            departure = m.addVars(index_1, name='departure')
            arrival = m.addVars(index_2, name='arrival')
            in_vehicle_j = m.addVars(index_3, name='in_vehicle_j')
            board = m.addVars(index_1, name='board')
            in_vehicle = m.addVars(index_4, name='in_vehicle')
            phi = m.addVars(index_1, name='phi')
            alight = m.addVars(index_4, name='alight')
            w = m.addVars(index_1, name='w')

            # add intermediate variables
            inter_board_limit_1=m.addVars(range(1,self.M+1),lb=-GRB.INFINITY,name="inter_board_limit_1")
            inter_board_limit = m.addVars(index_2,lb=-GRB.INFINITY, name="inter_board_limit")

            tau = m.addVars(index_2, name='tau')
            #in_vehicle_waiting = m.addVar(name='in_vehicle_waiting')
            #at_stop_waiting = m.addVar(name='at_stop_waiting')
            #extra_waiting = m.addVar(name='extra_waiting')

            #add holding time
            holding_time=m.addVars(index_2,name='holding_time')

            m.update()
            # lhs=LinExpr(0)
            item1 = gp.quicksum(
                (departure[x, y] - departure[x - 1, y]) * (departure[x, y] - departure[x - 1, y]) * self.lambda_[
                    y - 1] / 2 for x in range(2, self.M + 1) for y in range(1, self.N + 1))
            item1 = item1 + gp.quicksum(
                self.lambda_[y - 1] / 2 * departure[1, y] * departure[1, y] for y in range(1, self.N + 1))
            item1 = item1 + gp.quicksum(self.lambda_[y - 1] / 2 * (self.t_bar[y - 1] - departure[self.M, y]) * (
                    self.t_bar[y - 1] - departure[self.M, y]) for y in range(1, self.N + 1))
            #item1 = item1 + gp.quicksum(self.lambda_[y - 1] / 2 * (t_bar[y ] - departure[self.M, y]) * (
             #          t_bar[y ] - departure[self.M, y]) for y in range(1, self.N + 1))

            # item2 = gp.quicksum(
            #    in_vehicle[x, y + 1] * tau[x, y] for x in range(1, self.M + 1) for y in range(2, self.N + 1))

            item2 = gp.quicksum(
                (in_vehicle[x, y + 1] - in_vehicle_j.prod(self.p, x, '*', y + 1)) * tau[x, y + 1] for x in
                range(1, self.M + 1) for y in range(1, self.N))

            item3 = gp.quicksum(
                w[x, y] * (departure[x + 1, y] - departure[x, y]) for x in range(1, self.M) for y in
                range(1, self.N + 1))
            item3 = item3 + gp.quicksum(
                w[self.M, y] * (self.t_bar[y - 1] - departure[self.M, y]) for y in range(1, self.N + 1))
            #item3 = item3 + gp.quicksum(
             #   w[self.M, y] * (t_bar[y] - departure[self.M, y]) for y in range(1, self.N + 1))

            m.setObjective(self.theta[0] * item1 + self.theta[1] * item2 + self.theta[2] * item3, sense=gp.GRB.MINIMIZE)

            #m.addConstr(in_vehicle_waiting == item2, name='in_vehicle_c')
            #m.addConstr(at_stop_waiting == item1, name='at_stop_c')
            #m.addConstr(extra_waiting == item3, name='extra_c')

            #m.addConstrs((holding_time[x,1] for x in range(1,self.M+1)),name='holding_time_1')
            m.addConstrs((departure[x, 1] == self.headway * x for x in range(1, self.M + 1)), name='depart_1')
            m.addConstrs(
                (departure[x, y] == departure[x, y - 1] + self.ll[y - 2] / self.v[y - 2] + tau[x, y] for x, y in
                 index_2), name='depart')
            m.addConstrs((arrival[x, y] == departure[x, y - 1] + self.ll[y - 2] / self.v[y - 2] for x, y in index_2),
                         name='arri')
            m.addConstrs(
                (in_vehicle_j[x, y, z] == in_vehicle_j[x, y, z - 1] * (1 - self.p[x, y, z - 1]) for x, y, z in index_3
                 if y != z - 1), name='in_j')
            m.addConstrs((in_vehicle_j[x, y, z] == board[x, y] for x, y, z in index_3 if z == y + 1), name='in_')
            m.addConstrs((in_vehicle[x, y] == in_vehicle_j.sum(x, '*', y) for x, y in index_4), name='inTotal')
            #m.addConstrs((phi[1, y] == self.lambda_[y - 1] * departure[1, y] for y in range(1, self.N + 1)),
            #             name='waiting_1')
            m.addConstrs((phi[1, y] == self.lambda_[y - 1] * self.headway/2 for y in range(1, self.N + 1)),
                         name='waiting_1')
            #m.addConstrs((phi[1, y] == self.lambda_[y - 1] * self.headway/2 for y in range(1, self.N + 1)),
            #             name='waiting_1')
            m.addConstrs(
                (phi[x, y] == self.lambda_[y - 1] * (departure[x, y] - departure[x - 1, y]) + w[x - 1, y] for x, y in
                 index_1 if x != 1), name='waiting')
            m.addConstrs((alight[x, z] == in_vehicle_j.prod(self.p, x, '*', z) for x, z in index_4), name='al')

            #m.addConstrs((w[x, 1] - phi[x, 1] + self.capacity >= 0 for x in range(1, self.M + 1)), name='w_1')
            #m.addConstrs(
            #    (w[x, y] - phi[x, y] + (self.capacity - in_vehicle[x, y] + alight[x, y]) >= 0 for x, y in index_2),
            #   name='w_')
            #m.addConstrs(
            #    (w[x, y]>= 0 for x, y in index_1),
            #   name='w_')

            #print("OK1")
            m.addConstrs((inter_board_limit_1[x]==phi[x,1]-self.capacity for x in range(1,self.M+1)),name="inter_board_limit_1_Con")
            m.addConstrs((w[x, 1] == max_(0, inter_board_limit_1[x]) for x in range(1, self.M + 1)), name='modify_w_1')
            m.addConstrs((inter_board_limit[x, y] == phi[x, y] - (self.capacity - in_vehicle[x, y] + alight[x, y]) for x, y in index_2), name="inter_board_limit_Con")
            m.addConstrs((w[x, y] == max_(0, inter_board_limit[x, y]) for x, y in index_2), name='modify_w_')

            #m.addConstrs((w[x, 1] == 0 for x in range(1, self.M + 1)), name='modify_w_1')
            #m.addConstrs((w[x, y] == 0 for x, y in index_2), name='modify_w_')

        #    m.addConstrs((w[x, 1] >= 0 for x in range(1, self.M + 1)), name='modify_w_1')
        #    m.addConstrs((w[x, y] >= 0 for x, y in index_2), name='modify_w_')
            #print("OK3")
            m.addConstrs((departure[x, y] - departure[x - 1, y] >= 0 for x, y in index_1 if x != 1),
                         name='overtakeing_n_1')
            m.addConstrs((arrival[x, y] - arrival[x - 1, y] >= 0 for x, y in index_2 if x != 1), name='overtakeing_n_2')
            m.addConstrs((self.t_bar[y - 1] - departure[self.M, y] >= 0 for y in range(1, self.N + 1)), name='virtual')
            m.addConstrs((board[x, y] == phi[x, y] - w[x, y] for x, y in index_1), name='factual_board')
            #m.addConstrs((dwelling_time[x,y]==max_(board[x,y],alight[x,y]) for x,y in index_2),name='actual_passenger')
            m.addConstrs((tau[x, y] == board[x, y] * self.boarding_rate / 60+holding_time[x,y] for x, y in index_2), name='duration')
            #m.addConstrs((tau[x, y] == dwelling_time[x, y] * self.boarding_rate / 60 for x, y in index_2), name='duration')

            # m.addConstr(t_bar[1]==self.headway*(self.M+1),name='t_depart_1')
            # m.addConstrs((t_bar[y]==t_bar[y-1]+self.ll[y-2]/self.v[y-2]+tau_bar[y] for y in range(2,self.N+1)),name='t_depart')
            # m.addConstrs((arrival_bar[y]==t_bar[y-1]+self.ll[y-2]/self.v[y-2] for y in range(2,self.N+1)),name='t_arrival')
            # m.addConstrs((arrival_bar[y]-arrival[self.M,y]>=0 for y in range(2,self.N+1)),name='t_overtaking')
            # m.addConstrs((tau_bar[y]==w[self.M,y]*self.boarding_rate/60 for y in range(2,self.N+1)),name='t_boarding')

            #m.addConstrs((t_bar[y]-departure[self.M,y]>=0 for y in range(1,self.N+1)),name='virtual')
            #print("OK4")
            m.optimize()
            #print(m.getAttr('x', w))
            if m.status == GRB.OPTIMAL:
                print(m.status)
                self._objVal = m.objVal
                #self._result = m.getAttr('x', [in_vehicle_waiting, at_stop_waiting, extra_waiting])
                self._departure = m.getAttr('x', departure)
                self._arrival = m.getAttr('x', arrival)
                self._in_vehicle_j = m.getAttr('x', in_vehicle_j)
                self._in_vehicle = m.getAttr('x', in_vehicle)
                self._board = m.getAttr('x', board)
                self._w = m.getAttr('x', w)
                self._phi = m.getAttr('x', phi)
                self._tau = m.getAttr('x', tau)
                self._alight = m.getAttr('x', alight)
                self._holding_time=m.getAttr('x',holding_time)

                self._item1 = gp.quicksum(
                    (self._departure[x, y] - self._departure[x - 1, y]) * (self._departure[x, y] - self._departure[x - 1, y]) * self.lambda_[
                        y - 1] / 2 for x in range(2, self.M + 1) for y in range(1, self.N + 1))
                self._item1 = self._item1 + gp.quicksum(
                    self.lambda_[y - 1] / 2 * self._departure[1, y] * self._departure[1, y] for y in range(1, self.N + 1))
                self._item1 = self._item1 + gp.quicksum(self.lambda_[y - 1] / 2 * (self.t_bar[y - 1] - self._departure[self.M, y]) * (
                        self.t_bar[y - 1] - self._departure[self.M, y]) for y in range(1, self.N + 1))
                self._item2 = gp.quicksum(
                    (self._in_vehicle[x, y + 1] - self._in_vehicle_j.prod(self.p, x, '*', y + 1)) * self._tau[x, y + 1] for x in
                    range(1, self.M + 1) for y in range(1, self.N))
                self._item3 = gp.quicksum(
                    self._w[x, y] * (self._departure[x + 1, y] - self._departure[x, y]) for x in range(1, self.M) for y in
                    range(1, self.N + 1))
                self._item3 = self._item3 + gp.quicksum(
                    self._w[self.M, y] * (self.t_bar[y - 1] - self._departure[self.M, y]) for y in range(1, self.N + 1))
                self._result = [self._item2.getValue(),self._item1.getValue(),self._item3.getValue()]
                # self.t_bar=m.getAttr('x',self.t_bar)
                #print(self.departure)
                #print(self.arrival)
                #print(self.in_vehicle_j)
                #print(self.in_vehicle)
                #print('board:', self.board)
                #print(self.w)
                #print(self.phi)
                #print(self.tau)
                #print('alight:', self.alight)
                ## print(self.t_bar)
                ## return self.result, self.departure, self.arrival, self.in_vehicle, self.w
                ## return self.objVal,self.result,self.departure,self.arrival,self.in_vehicle_j,self.in_vehicle,self.board,self.w,self.phi,self.tau,self.alight
            elif m.status == GRB.TIME_LIMIT:
                # m.Params.timeLimit=float("inf")
                m.Params.timeLimit = 200
                if m.MIPGap <= 0.05:
                    print(m.status)
                    print(m.MIPGap)
                    self._objVal = m.objVal
                    #self._result = m.getAttr('x', [in_vehicle_waiting, at_stop_waiting, extra_waiting])
                    self._departure = m.getAttr('x', departure)
                    self._arrival = m.getAttr('x', arrival)
                    self._in_vehicle_j = m.getAttr('x', in_vehicle_j)
                    self._in_vehicle = m.getAttr('x', in_vehicle)
                    self._board = m.getAttr('x', board)
                    self._w = m.getAttr('x', w)
                    self._phi = m.getAttr('x', phi)
                    self._tau = m.getAttr('x', tau)
                    self._alight = m.getAttr('x', alight)
                    self._holding_time = m.getAttr('x', holding_time)

                    self._item1 = gp.quicksum(
                        (self._departure[x, y] - self._departure[x - 1, y]) * (
                                    self._departure[x, y] - self._departure[x - 1, y]) * self.lambda_[
                            y - 1] / 2 for x in range(2, self.M + 1) for y in range(1, self.N + 1))
                    self._item1 = self._item1 + gp.quicksum(
                        self.lambda_[y - 1] / 2 * self._departure[1, y] * self._departure[1, y] for y in
                        range(1, self.N + 1))
                    self._item1 = self._item1 + gp.quicksum(
                        self.lambda_[y - 1] / 2 * (self.t_bar[y - 1] - self._departure[self.M, y]) * (
                                self.t_bar[y - 1] - self._departure[self.M, y]) for y in range(1, self.N + 1))
                    self._item2 = gp.quicksum(
                        (self._in_vehicle[x, y + 1] - self._in_vehicle_j.prod(self.p, x, '*', y + 1)) * self._tau[
                            x, y + 1] for x in
                        range(1, self.M + 1) for y in range(1, self.N))
                    self._item3 = gp.quicksum(
                        self._w[x, y] * (self._departure[x + 1, y] - self._departure[x, y]) for x in range(1, self.M)
                        for y in
                        range(1, self.N + 1))
                    self._item3 = self._item3 + gp.quicksum(
                        self._w[self.M, y] * (self.t_bar[y - 1] - self._departure[self.M, y]) for y in
                        range(1, self.N + 1))
                    self._result = [self._item2.getValue(), self._item1.getValue(), self._item3.getValue()]
                    # self.t_bar=m.getAttr('x',self.t_bar)
                    #print(self.departure)
                    #print(self.arrival)
                    #print(self.in_vehicle_j)
                    #print(self.in_vehicle)
                    #print('board:', self.board)
                    #print(self.w)
                    #print(self.phi)
                    #print(self.tau)
                    #print('alight:', self.alight)
                else:
                    m.Params.MIPGap = 0.05
                    m.optimize()
                    print("OK")
                    print(m.status)
                    self._objVal = m.objVal
                    #self._result = m.getAttr('x', [in_vehicle_waiting, at_stop_waiting, extra_waiting])
                    self._departure = m.getAttr('x', departure)
                    self._arrival = m.getAttr('x', arrival)
                    self._in_vehicle_j = m.getAttr('x', in_vehicle_j)
                    self._in_vehicle = m.getAttr('x', in_vehicle)
                    self._board = m.getAttr('x', board)
                    self._w = m.getAttr('x', w)
                    self._phi = m.getAttr('x', phi)
                    self._tau = m.getAttr('x', tau)
                    self._alight = m.getAttr('x', alight)
                    self._holding_time = m.getAttr('x', holding_time)

                    self._item1 = gp.quicksum(
                        (self._departure[x, y] - self._departure[x - 1, y]) * (
                                    self._departure[x, y] - self._departure[x - 1, y]) * self.lambda_[
                            y - 1] / 2 for x in range(2, self.M + 1) for y in range(1, self.N + 1))
                    self._item1 = self._item1 + gp.quicksum(
                        self.lambda_[y - 1] / 2 * self._departure[1, y] * self._departure[1, y] for y in
                        range(1, self.N + 1))
                    self._item1 = self._item1 + gp.quicksum(
                        self.lambda_[y - 1] / 2 * (self.t_bar[y - 1] - self._departure[self.M, y]) * (
                                self.t_bar[y - 1] - self._departure[self.M, y]) for y in range(1, self.N + 1))
                    self._item2 = gp.quicksum(
                        (self._in_vehicle[x, y + 1] - self._in_vehicle_j.prod(self.p, x, '*', y + 1)) * self._tau[
                            x, y + 1] for x in
                        range(1, self.M + 1) for y in range(1, self.N))
                    self._item3 = gp.quicksum(
                        self._w[x, y] * (self._departure[x + 1, y] - self._departure[x, y]) for x in range(1, self.M)
                        for y in
                        range(1, self.N + 1))
                    self._item3 = self._item3 + gp.quicksum(
                        self._w[self.M, y] * (self.t_bar[y - 1] - self._departure[self.M, y]) for y in
                        range(1, self.N + 1))
                    self._result = [self._item2.getValue(), self._item1.getValue(), self._item3.getValue()]
                    # self.t_bar=m.getAttr('x',self.t_bar)
                    #print(self.departure)
                    #print(self.arrival)
                    #print(self.in_vehicle_j)
                    #print(self.in_vehicle)
                    #print('board:', self.board)
                    #print(self.w)
                    #print(self.phi)
                    #print(self.tau)
                    #print('alight:', self.alight)
            return self._objVal, self._result, self._departure, self._arrival, self._in_vehicle_j, self._in_vehicle, self._board, self._w, self._phi, self._tau, self._alight,self._holding_time

        except gp.GurobiError as e:
            print('Error code' + str(e.errno) + ': ' + str(e))
        except AttributeError:
            print('Encountered an attribute error')

    def __call__(self,*args,**kwargs):
        results=self.__Optimal()
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


    def Analysis(self):
        self.__Optimal()
        folder=self.directory('Results_ZC')

        result=np.array(self._result).reshape(1,len(self._result))
        data_result=pd.DataFrame(result,columns=["In-vehicle","At-stop","Extra"],index=["Holding control"])
        data_result['Total']=data_result.apply(lambda x: x.sum(),axis=1)
        filedata_result_csv=self.combine_path(folder,'results','csv')
        #data_result.to_csv('Results/results.csv',mode="a+")
        data_result.to_csv(filedata_result_csv, mode="a+")
        filedata_result_excel = self.combine_path(folder, 'results', 'xlsx')
        #data_result.to_excel('Results/results.xlsx',sheet_name="Sheet1")
        data_result.to_excel(filedata_result_excel, sheet_name="Sheet2")

        print(self._in_vehicle.select(1,'*'))
        def generate_in_vehicle(item:list):
            temp=[0]
            a=copy.deepcopy(item)
            temp.extend(a)
            return temp
        data_in_vehicle={str(i):generate_in_vehicle(self._in_vehicle.select(i,'*')) for i in range(1,self.M+1)}
        data_in_vehicle=pd.DataFrame(data_in_vehicle,index=range(1,self.N+2))
        #print(data_in_vehicle.head())
        print(data_in_vehicle)
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
        filename_svg=self.combine_path(folder,"Passenger Loads under Holding Control","svg")
        #plt.savefig('Results/Average passenger arrival rates at each bus stations.svg')
        plt.savefig(filename_svg,bbox_extra_artists=(lg,),bbox_inches='tight')
        filename_pdf = self.combine_path(folder, "Passenger Loads under Holding Control", "pdf")
        #plt.savefig('Results/Average passenger arrival rates at each bus stations.pdf',dpi=1000)
        plt.savefig(filename_pdf, dpi=1000,bbox_extra_artists=(lg,),bbox_inches='tight')


        #prepare data
        data_passenger=[[i,j,self._board.select(i,j)[0],self._w.select(i,j)[0],self._phi.select(i,j)[0]] for i in range(1,self.M+1) for j in range(1,self.N+1)]
        data_passenger=pd.DataFrame(data_passenger,columns=['Bus','Stop','Boarding','Still need to wait','Total wait'])
        #print(data_passenger)
        x_var='Stop'
        groupby_var='Bus'
        data_b=data_passenger.groupby(groupby_var)
        bar_x=np.arange(1,self.N+1)
        bar_width=1/(self.M+1)
        bar_tick_label=list(map(lambda x:str(x),bar_x))
        colors = [plt.cm.Spectral(i / float(2 - 1)) for i in range(2)]
        #colors=['#8E354A','#261E47']
        plt.figure(num=2, facecolor='white', edgecolor='black')
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
        filename_svg = self.combine_path(folder, "About to board and stranded passengers because of capacity limit under Holding Control", "svg")
        plt.savefig(filename_svg)
        filename_pdf = self.combine_path(folder, "About to board and stranded passengers because of capacity limit under Holding Control", "pdf")
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
        plt.figure(num=3, facecolor='white', edgecolor='black')
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
        filename_svg = self.combine_path(folder, "Average number of about to board and stranded passengers because of capacity limit under holding control",
                                         "svg")
        plt.savefig(filename_svg)
        filename_pdf = self.combine_path(folder, "Average about to board and stranded passengers because of capacity limit under holding control",
                                         "pdf")
        plt.savefig(filename_pdf, dpi=1000)

        #route trajectory
        data_trajectory=[[self._departure.select(i,1)[0]] for i in range(1,self.M+1)]
        [data_trajectory[i-1].extend([self._arrival.select(i,j)[0],self._departure.select(i,j)[0]]) for i in range(1,self.M+1) for j in range(2,self.N+1)]
        [data_trajectory[i-1].extend([self._departure.select(i,self.N)[0]+self.ll[self.N-1]/self.v[self.N-1]]) for i in range(1,self.M+1)]
        data_trajectory=np.array(data_trajectory).transpose()
        column_name=list(map(str,range(1,self.M+1)))
        data_trajectory=pd.DataFrame(data_trajectory,columns=column_name)
        temp_list=list(zip(range(1,self.N+1),range(1,self.N+1)))
        temp_list=[list(i) for i in temp_list]
        temp_list=list(reduce(lambda x,y:x+y,temp_list,[]))
        temp_list.pop(0)
        temp_list.append(self.N+1)
        #print(temp_list)
        data_trajectory['Stop']=temp_list
        print(data_trajectory)
        #print(type(data_trajectory))
        plt.figure(num=4, facecolor='white', edgecolor='black')
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
        filename_svg = self.combine_path(folder, "Bus Trajectory under Holding Control", "svg")
        plt.savefig(filename_svg,bbox_extra_artists=(lg,),bbox_inches='tight')
        filename_pdf = self.combine_path(folder, "Bus Trajectory under Holding Control", "pdf")
        plt.savefig(filename_pdf, dpi=1000,bbox_extra_artists=(lg,),bbox_inches='tight')