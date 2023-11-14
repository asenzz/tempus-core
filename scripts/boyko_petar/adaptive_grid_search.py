# factor of MSE ratio to steps count dilation/contraction
from google.protobuf.descriptor import DescriptorBase

Default_step_degree = 0.8

# number of steps per training a range of parameter values
Default_num_steps = 7

# uncoditionally multiply steps count every recursion
Default_step_coef = 0.5

# default data chunk lenght in number of instances
Default_chunk_len = 128



# ANSI CSI bold print constant
BOLDTEXT_ = "\x1b[1m"

# ANSI CSI plain print constant
PLAINTEXT_ = "\x1b[0m"

# enable debug code
DEBUG_ = True


# diapason for values
# C, epsilon, kernel_param, kernel_param2
low = (1, 1e-3, 1, 1)
up = (100, 0.1, 1000, 1000)

Db_prepare_cmd = "echo f|sudo -S /home/mts/Projects/asenzett-code/dbscripts/replace_main_schema.sh > /dev/null 2>&1"
Program_path = "/home/mts/Projects/asenzett-code/build/SVRMain"
Program_params = "setUser=1\n\
user-id=1\n\
user-name=svrwave\n\
user-email=svrwave@google.com\n\
user-password=svrwave\n\
\n\
setDataset=1\n\
entity=0\n\
dataset-name=Dataset1\n\
swtLevels=4\n\
wavelet-name=haar\n\
lookback-time=5\n\
\n\
setQueue=1\n\
queue-name=Queue1\n\
resolution=60\n\
deviation=5\n\
value-columns=val1 val2 val3 val4\n\
table-name=Table1\n\
data-filename=/home/mts/Projects/temp/EURUSDpro1.csv\n\
"



import numpy as np
import subprocess
import os.path
from os import remove
import re
import sys
import time

def LOG(_log_data):
    print time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), " ", _log_data

def svr_train(
        _svr_params     # list of svr parameters: C, epsilon, kernel_param, kernel_param2
):
    dir_path = os.path.dirname(Program_path)
    os.chdir(dir_path)
    command_path = dir_path + str(int(time.time()))
    file = open(command_path, "w")
    file.write(Program_params)
    params_str = ""
    for parval in _svr_params:
        params_str += "%s " % str(parval)
    file.write("svrParameters=%s1 1.5\n" % params_str)
    file.close()
    cmd = [Program_path,
           "--optionsFile=%s" % command_path]
    subprocess.check_call(Db_prepare_cmd, shell=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    remove(command_path)

    res_str = re.findall("] INFO:\s+mse \d+\.\d+", out)
    assert (len(res_str) == 1),"Can't find mse is output!!!"
    result = float(re.findall("\d+\.\d+", res_str[0])[0])
    return result



# number of train calls during optimization
train_cycles = 0

"""
Train cycle implementing sort of stochastic gradient descent in a recursive
 fashion. It is used during the SVM parameters optimization process.
"""
def trainrep(
    _svr_params,    # list of svr parameters: C, epsilon, kernel_param, kernel_param2
    _begin,
    _end,
    _num_steps=Default_chunk_len,
    _prev_best_mse=0,
    _train_param=None,
    ):

    # stop optimizing when number of sub steps is less than one
    if _num_steps < 1: return _prev_best_mse

    # # some sanity checking
    if _end < low[_train_param]:
        _end == low[_train_param]
    if _begin < low[_train_param]:
        _begin = low[_train_param]


    # absolute counter of svm_train calls
    global train_cycles
    # # put a dot on show and flush immediately
    # sys.stdout.write(".")
    # sys.stdout.flush()

    # if previous MSE is 0 do one train and get mean squared error
    if _prev_best_mse == 0:
        LOG("initial " + str(_svr_params))
        _prev_best_mse = svr_train(_svr_params)
        LOG("mse = " + str(_prev_best_mse))

    # save initial values
    best_mse = _prev_best_mse
    best_parval = _svr_params[_train_param]
    step_size = float(_end - _begin) / _num_steps

    # number of steps to be used for self calls
    num_sub_steps = _num_steps * Default_step_coef

    LOG(str(num_sub_steps) + " in (" + str(_begin) + ", " + str(_end) + ")")

    # do a grid search in the range starting from _being to _end in _num_steps count
    for parval in np.linspace(_begin, _end, _num_steps):

        # bump up counter
        train_cycles += 1

        # set optimized parameter to value step
        _svr_params[_train_param] = parval

        LOG(str(train_cycles) + " " + str(_svr_params))

        # get MSE for current parameters
        mse = svr_train(_svr_params)
        LOG("mse = " + str(mse))

        # see if we got something better and save it for return
        # if we got zero MSE then break out of the loop immediately
        if best_mse > mse:
            best_mse = mse
            best_parval = _svr_params[_train_param]
            LOG("changing best_svr_params " + str(_svr_params))
            if mse == 0: break

        # call self with current best MSE and same optimized parameter as inputs
        # range is of tweaking is set to the begin of previous step and end on the next one.
        # number of steps to be used set by multiplying ratio of this MSE to current best MSE
        mse = trainrep(
            _svr_params,
            _begin=parval - step_size,
            _end=parval + step_size,
            _num_steps=num_sub_steps * ((best_mse / mse) ** Default_step_degree),
            _prev_best_mse=best_mse,
            _train_param=_train_param)

    # set best parameter value and best MSE
    _svr_params[_train_param] = best_parval
    return best_mse


"""
Given a SVM problem class and a SVM parameters class tweak parameter
values for best performance given the problem set. It's best to use
at least 2 cycles of optimization and more than 5 steps.
"""
def optimize_svr_params(
    _svr_param, 				# tweak this svm_parameter object or create new one
    _num_cycles=2, 			# number of repetitions to do on all tweaked parameters
    _num_steps=Default_num_steps	# starting number of steps per training cycle, decreases by __steps_coef every level
    ):

    # check input sanity
    if _num_steps < 1 or _num_cycles < 1: return -1

    # init locals and include global variable train_cycles for updating and print
    global train_cycles
    mse = 0
    train_cycles = 0

    # # set epsilon based on LibSVM user guide recommendation and a bit lower
    # # WARNING: going further will probably make training take eternally
    # if _param.svm_type in [NU_SVC, NU_SVR]: _param.eps = 1e-10
    # else: _param.eps = 1e-5

    # start num_cycles optimizations of all used parameters
    # for the specified SVM type and kernel
    for ix in range(_num_cycles):
        print BOLDTEXT_,"Optimization run",ix+1

        LOG("Optimizing parameter C ")
        mse = trainrep(_svr_param, low[0], up[0], _num_steps, mse, 0)
        LOG("Optimized to " + str(_svr_param))

        LOG("Optimizing parameter Epsilon")
        mse = trainrep(_svr_param, 0.005, up[1], _num_steps, mse, 1)
        LOG("Optimized to " + str(_svr_param))

        LOG("Optimizing parameter kernel_param ")
        mse = trainrep(_svr_param, low[2], up[2], _num_steps, mse, 2)
        LOG("Optimized to " + str(_svr_param))

        LOG("Optimizing parameter kernel_param2 ")
        mse = trainrep(_svr_param, low[3], up[3], _num_steps, mse, 3)
        LOG("Optimized to " + str(_svr_param))



    # # set epsilon based on LibSVM user guide recommendation and a bit lower
    # # WARNING: going further will probably make training take eternal time
    # if _param.svm_type in [NU_SVC, NU_SVR]: _param.eps = 1e-12
    # else: _param.eps = 1e-7


    # Print results
    print PLAINTEXT_,"Parameter optimization done in ",train_cycles," train cycles. "
    print "Mean squared error ",mse
    if DEBUG_:
        print "SVM parameters follow (C, epsilon, kernel_param, kernel_param2) :"
        print _svr_param
        print '\n'
    return _svr_param

"""
Main function as prescribed by Guido van Rossum.
"""
def main():
    svr_param = [10, 0.01, 17, 30]
    # [10, 0.16666708333333333, 17, 30]
    print optimize_svr_params(svr_param)
    # print svr_train(svr_param)
"""
Call main function only if we're invoked from this program.
"""
if __name__ == "__main__":  main()