import gurobipy as gp
import numpy as np
import scipy.stats as st
from gurobipy import *


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
        self._lambda_ = [0.6, 0.7, 0.75, 1, 0.8, 0.5, 1.2, 1.5, 1.3, 1.6, 2, 2, 3, 1.2, 2.2, 2, 1.2, 0.8, 1, 0.8,
                         0.7, 0.6, 0.5, 0.5]
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
        temp = [(x, y, z) for x in range(1, self.M + 1) for y in range(1, self.N + 1) for z in range(y + 1, self.N + 2)]
        temp = tuplelist(temp)

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
        self._headway = 6
        return self._headway

    @headway.setter
    def headway(self, value):
        self._headway = value

    @property
    def boarding_rate(self):  # s/pax
        self._boarding_rate = 2
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

    def Optimal(self):
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

            #t_bar = m.addVars(range(1, self.N + 1), name='t_tar')

            # arrival_bar=m.addVars(range(2,self.N+1),name='arrival_bar')
            # tau_bar=m.addVars(range(2,self.N+1),name='tau_bar')

            # add intermediate variables
            inter_board_limit_1=m.addVars(range(1,self.M+1),lb=-GRB.INFINITY,name="inter_board_limit_1")
            inter_board_limit = m.addVars(index_2,lb=-GRB.INFINITY, name="inter_board_limit")

            tau = m.addVars(index_2, name='tau')
            in_vehicle_waiting = m.addVar(name='in_vehicle_waiting')
            at_stop_waiting = m.addVar(name='at_stop_waiting')
            extra_waiting = m.addVar(name='extra_waiting')

            #holding_time=m.addVars(index_2,name='holding_time')

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

            m.addConstr(in_vehicle_waiting == item2, name='in_vehicle_c')
            m.addConstr(at_stop_waiting == item1, name='at_stop_c')
            m.addConstr(extra_waiting == item3, name='extra_c')

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
            m.addConstrs((tau[x, y] == board[x, y] * self.boarding_rate / 60 for x, y in index_2), name='duration')

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
                self.objVal = m.objVal
                self.result = m.getAttr('x', [in_vehicle_waiting, at_stop_waiting, extra_waiting])
                self.departure = m.getAttr('x', departure)
                self.arrival = m.getAttr('x', arrival)
                self.in_vehicle_j = m.getAttr('x', in_vehicle_j)
                self.in_vehicle = m.getAttr('x', in_vehicle)
                self.board = m.getAttr('x', board)
                self.w = m.getAttr('x', w)
                self.phi = m.getAttr('x', phi)
                self.tau = m.getAttr('x', tau)
                self.alight = m.getAttr('x', alight)
                # self.t_bar=m.getAttr('x',self.t_bar)
                print(self.departure)
                print(self.arrival)
                print(self.in_vehicle_j)
                print(self.in_vehicle)
                print('board:', self.board)
                print(self.w)
                print(self.phi)
                print(self.tau)
                print('alight:', self.alight)
                # print(self.t_bar)
                # return self.result, self.departure, self.arrival, self.in_vehicle, self.w
                # return self.objVal,self.result,self.departure,self.arrival,self.in_vehicle_j,self.in_vehicle,self.board,self.w,self.phi,self.tau,self.alight
            elif m.status == GRB.TIME_LIMIT:
                # m.Params.timeLimit=float("inf")
                m.Params.timeLimit = 200
                if m.MIPGap <= 0.05:
                    print(m.status)
                    print(m.MIPGap)
                    self.objVal = m.objVal
                    self.result = m.getAttr('x', [in_vehicle_waiting, at_stop_waiting, extra_waiting])
                    self.departure = m.getAttr('x', departure)
                    self.arrival = m.getAttr('x', arrival)
                    self.in_vehicle_j = m.getAttr('x', in_vehicle_j)
                    self.in_vehicle = m.getAttr('x', in_vehicle)
                    self.board = m.getAttr('x', board)
                    self.w = m.getAttr('x', w)
                    self.phi = m.getAttr('x', phi)
                    self.tau = m.getAttr('x', tau)
                    self.alight = m.getAttr('x', alight)
                    # self.t_bar=m.getAttr('x',self.t_bar)
                    print(self.departure)
                    print(self.arrival)
                    print(self.in_vehicle_j)
                    print(self.in_vehicle)
                    print('board:', self.board)
                    print(self.w)
                    print(self.phi)
                    print(self.tau)
                    print('alight:', self.alight)
                else:
                    m.Params.MIPGap = 0.05
                    m.optimize()
                    print("OK")
                    print(m.status)
                    self.objVal = m.objVal
                    self.result = m.getAttr('x', [in_vehicle_waiting, at_stop_waiting, extra_waiting])
                    self.departure = m.getAttr('x', departure)
                    self.arrival = m.getAttr('x', arrival)
                    self.in_vehicle_j = m.getAttr('x', in_vehicle_j)
                    self.in_vehicle = m.getAttr('x', in_vehicle)
                    self.board = m.getAttr('x', board)
                    self.w = m.getAttr('x', w)
                    self.phi = m.getAttr('x', phi)
                    self.tau = m.getAttr('x', tau)
                    self.alight = m.getAttr('x', alight)
                    # self.t_bar=m.getAttr('x',self.t_bar)
                    print(self.departure)
                    print(self.arrival)
                    print(self.in_vehicle_j)
                    print(self.in_vehicle)
                    print('board:', self.board)
                    print(self.w)
                    print(self.phi)
                    print(self.tau)
                    print('alight:', self.alight)
            return self.objVal, self.result, self.departure, self.arrival, self.in_vehicle_j, self.in_vehicle, self.board, self.w, self.phi, self.tau, self.alight


        except gp.GurobiError as e:
            print('Error code' + str(e.errno) + ': ' + str(e))
        except AttributeError:
            print('Encountered an attribute error')

    
