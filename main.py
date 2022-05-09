#from Model1 import *
import Model1 as mb
import Model1_holding as mh
import CollaborativeScenario as c
import CollaborativeScenario_holding as ch
import logging
import time
#import datetime
import traceback
import operator
from functools import reduce
if __name__=="__main__":
    #benchmark
    '''
    zc=mb.Model1(M=8,N=20)
    #a=zc.Optimal()
    #a=zc()
    #print(a)
    zc.Analysis()
    '''
    #holding control
    '''
    zc=mh.Model1(M=8,N=20)
    zc.Analysis()    
    '''
    #freight transport
    '''
    zc=c.Collaborative(M=3,N=3)
    #a,b,c,d=zc.demand_parcels()
    #print(d)
    #zc.Optimal(1, current_data=(0,0,0))
    #zc_database,zc_df=zc.dynamic_programming()
    #print(zc_database)
    #print(zc_df)
    #zc_df.to_csv('test_DP.csv')
    zc.Analysis()
    #zc.Average_Analysis()
    '''
    #freight transport under holding control
    '''
    zc=ch.Collaborative(M=3,N=3)
    #zc.Analysis()
    zc.Average_Analysis()
    '''

    logging.basicConfig(filename='log.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    total_time=[]
    # no control scenario
    t_start_no_control = time.perf_counter_ns()
    try:
        logging.info('Try no control scenario...')
        no_control = mb.Model1(M=8, N=20)
        no_control.Analysis()
        logging.info('No control scenario complete!')
    except Exception as e:
        logging.debug('Except: {}'.format(e))
        logging.debug(traceback.format_exc)
    finally:
        t_no_control=time.perf_counter_ns()
        t_no_control_duration=(t_no_control-t_start_no_control)/1000/1000
        total_time.extend([t_no_control_duration])
        logging.info('Finally...')
        logging.info('Execution time of no control: %s' % ((t_no_control-t_start_no_control)/1000/1000)) #micro seconds
        logging.info('End')

    # holding control scenario
    t_start_holding=time.perf_counter_ns()
    try:
        logging.info('Try holding control scenario...')
        holding_control = mh.Model1(M=8, N=20)
        holding_control.Analysis()
        logging.info('Holding control scenario complete!')
    except Exception as e:
        logging.debug('Except: {}'.format(e))
        logging.debug(traceback.format_exc())
    finally:
        t_holding_control=time.perf_counter_ns()
        t_holding_control_duration=(t_holding_control-t_start_holding)/1000/1000
        total_time.extend([t_holding_control_duration])
        logging.info('Finally...')
        logging.info('Execution time of holding control: %s' % ((t_holding_control-t_start_holding)/1000/1000)) #micro seconds
        logging.info('End')

    demand_list = [5, 10, 15, 20]

    # FTNC scenario
    t_start_FTNC=time.perf_counter_ns()
    try:
        logging.info('Try FTNC scenario...')
        FTNC = c.Collaborative(M=8, N=20)
        FTNC.Analysis()
        logging.info('FTNC scenario complete!')
    except Exception as e:
        logging.debug('Except: {}'.format(e))
        logging.debug(traceback.format_exc())
    finally:
        t_FTNC=time.perf_counter_ns()
        t_FTNC_duration=(t_FTNC-t_start_FTNC)/1000/1000
        total_time.extend([t_FTNC_duration])
        logging.info('Finally...')
        logging.info('Execution time of FTNC: %s' % ((t_FTNC-t_start_FTNC)/1000/1000)) #micro seconds
        logging.info('End')
    #demand_list=[5, 10, 15, 20]
    t_start_FTNC_average=time.perf_counter_ns()

    def cal_exec_time(object, demand, t_start):
        object.demand=demand
        try:
            logging.info('Try average FTNC scenario, when demand is %s' % demand)
            object.Average_Analysis()
            logging.info('Average FTNC scenario complete, when demand is %s' % demand)
        except Exception as e:
            logging.info('Except: {}'.format(e))
            logging.debug(traceback.format_exc())
        finally:
            t_end=time.perf_counter_ns()
            t_duration=(t_end-t_start)/1000/1000
            logging.info('Finally...')
            logging.info('Execution time of average FTNC when demand is {} is {}'.format(demand,t_duration)) #micro seconds
            logging.info('End')
        return t_duration

    exec_time=[cal_exec_time(FTNC,i,t_start_FTNC_average) for i in demand_list]

    def myiter_sub(iterable,*,initial=None):
        it=iter(iterable)
        total=initial
        if initial==None:
            try:
                total=next(it)
            except StopIteration:
                return
        yield total
        for element in it:
            total=element-total
            yield total
            total=element

    exec_avg_time_FTNC=list(myiter_sub(exec_time))
    total_avg_FTNC=reduce(operator.add,exec_avg_time_FTNC)
    logging.info('Execution time of average FTNC is {}'.format(exec_avg_time_FTNC))  # micro seconds
    logging.info('Total execution time of average FTNC is {}'.format(total_avg_FTNC))  # micro seconds
    total_time.extend(total_avg_FTNC)

    # FTHC scenario
    t_start_FTHC = time.perf_counter_ns()
    try:
        logging.info('Try FTHC scenario...')
        FTHC = ch.Collaborative(M=8, N=20)
        FTHC.Analysis()
        logging.info('FTHC scenario complete!')
    except Exception as e:
        logging.debug('Except: {}'.format(e))
        logging.debug(traceback.format_exc())
    finally:
        t_FTHC = time.perf_counter_ns()
        t_FTHC_duration=(t_FTHC-t_start_FTHC)/1000/1000
        total_time.extend([t_FTHC_duration])
        logging.info('Finally...')
        logging.info('Execution time of FTHC: %s' % ((t_FTHC - t_start_FTHC) / 1000 / 1000))  # micro seconds
        logging.info('End')
    # demand_list=[5, 10, 15, 20]
    t_start_FTHC_average = time.perf_counter_ns()

    def cal_exec_time_FTHC(object, demand, t_start):
        object.demand = demand
        try:
            logging.info('Try average FTHC scenario, when demand is %s' % demand)
            object.Average_Analysis()
            logging.info('Average FTHC scenario complete, when demand is %s' % demand)
        except Exception as e:
            logging.info('Except: {}'.format(e))
            logging.debug(traceback.format_exc())
        finally:
            t_end = time.perf_counter_ns()
            t_duration = (t_end - t_start) / 1000 / 1000
            logging.info('Finally...')
            logging.info('Execution time of average FTHC when demand is {} is {}'.format(demand, t_duration))  # micro seconds
            logging.info('End')
        return t_duration

    exec_time = [cal_exec_time_FTHC(FTHC, i, t_start_FTHC_average) for i in demand_list]

    def myiter_sub(iterable, *, initial=None):
        it = iter(iterable)
        total = initial
        if initial == None:
            try:
                total = next(it)
            except StopIteration:
                return
        yield total
        for element in it:
            total = element - total
            yield total
            total = element

    exec_avg_time_FTHC = list(myiter_sub(exec_time))
    total_avg_FTHC=reduce(operator.add,exec_avg_time_FTNC)
    logging.info('Execution time of average FTHC is {}'.format(exec_avg_time_FTHC))  # micro seconds
    logging.info('Total execution time of average FTHC is {}'.format(total_avg_FTHC))  # micro seconds
    total_time.extend(total_avg_FTHC)

    print('Total time: ',total_time)
    print('Average execution time of FTNC: ',exec_avg_time_FTNC)
    print('Average execution time of FTHC: ',exec_avg_time_FTHC)
    final_total_time=reduce(operator.add,total_time)
    print('Overall execution time is ', final_total_time)




