import tensorflow as tf
import tensorflow_probability as tfp
from advi.core import run_advi_old, run_advi
import datetime
import time


def run_train_advi(model, train_data, test_data,
                   step_limit=100, m=1, p=1, skip_steps=10, old=False):
    # set up joint_log_prob and log_likelihood with training and test data
    joint_log_prob2 = lambda *args: model.joint_log_prob(train_data, *args)
    avg_log_likelihood2 = lambda *args: model.avg_log_likelihood(test_data, *args)

    # set up trace function for advi
    def trace_fn(advi, step):
        #print(step)
        if step % skip_steps == 0:
            if(old==True):
                logger.log_step("elbo", advi.elbo(p), step)
                logger.log_step("avg log pred advi",
                                advi_to_avg_log_like(advi, avg_log_likelihood2, p), step)
            else:
                # run new function which does not re calc the elbo
                logger.log_step("elbo", advi.current_elbo, step)
                logger.log_step("avg log pred advi",
                                advi_to_avg_log_like(advi, avg_log_likelihood2, p), step)

    # run advi
    print("running ADVI")
    # set up logger and run chain
    filename = "./logs/{}_advi.csv".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger = Logger(filename)
    if(old==True):
        advi_res = run_advi_old(shape=model.num_params,
                                target_log_prob_fn=joint_log_prob2,
                                bijector=model.bijector(),
                                m=m,
                                step_limit=step_limit,
                                trace_fn=trace_fn)
    else:
        advi_res = run_advi(shape=model.num_params,
                            target_log_prob_fn=joint_log_prob2,
                            bijector=model.bijector(),
                            m=m,
                            step_limit=step_limit,
                            trace_fn=trace_fn)
        
    logger.close()
    print("advi done")

    return advi_res


def run_train_hmc(model, train_data, test_data, step_size,
                  num_results=100, num_burnin_steps=100, skip_steps=10):

    # trace_functions for hmc
    # this function operates at every step of the chain
    def trace_fn(state, results):
        #print("Step {}".format(results.step))
        if results.step % skip_steps == 0:
            logger.log_step("avg log pred hmc",
                            "{}".format(state_to_avg_log_like(state, test_data, model)),
                            results.step)
        return ()

    # set up joint_log_prob with training data
    joint_log_prob2 = lambda *args: model.joint_log_prob(train_data, *args)

    # set up initial chain state
    initial_chain_state = model.return_initial_state()

    # Defining kernel for HMC
    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=joint_log_prob2,
            num_leapfrog_steps=3,
            step_size=step_size),
        num_adaptation_steps=int(num_burnin_steps * 0.8))

    # Run HMC
    def run_chain_hmc():
        print("running hmc chain")
      # Run the chain (with burn-in).
        states, is_accepted = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=initial_chain_state,
            kernel=adaptive_hmc,
            trace_fn=trace_fn)
        return states, is_accepted

    # set up logger and run chain
    filename = "./logs/{}_hmc.csv".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger = Logger(filename)
    states, is_accepted = run_chain_hmc()
    logger.close()
    print("hmc done")

    return states, is_accepted


def run_train_nuts(model, train_data, test_data, step_size,
                   num_results=20, num_burnin_steps=0, skip_steps=1):

    # trace_functions for nuts
    # this function operates at every step of the chain
    def trace_fn_nuts(state, results):
        step = num_burnin_steps + logger.counter()
        #print(step)
        if step % skip_steps == 0:
            logger.log_step("avg log pred nuts",
                            "{}".format(state_to_avg_log_like(state, test_data, model)),
                            step)
        return ()

    # set up joint_log_prob with training data
    joint_log_prob2 = lambda *args: model.joint_log_prob(train_data, *args)

    # set up initial chain state
    initial_chain_state = model.return_initial_state()

    # Defining kernel for NUTS
    nuts = tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=joint_log_prob2,
        step_size=step_size,
        max_tree_depth=10,
        max_energy_diff=1000.0,
        unrolled_leapfrog_steps=1, parallel_iterations=10, seed=None,
        name=None)

    # Run NUTS
    def run_chain_nuts():
        print("running nuts chain")
        # Run the chain (with burn-in).
        states, pkr = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=initial_chain_state,
            kernel=nuts,
            trace_fn=trace_fn_nuts)
        return states, pkr

    # set up logger and run chain
    filename = "./logs/{}_nuts.csv".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger = Logger(filename)
    states, _ = run_chain_nuts()
    logger.close()
    print("nuts done")

    return states


def state_to_avg_log_like(states, data, model):
    sample_mean = tf.reduce_mean(states, axis=[0])
    return model.avg_log_likelihood(data, states)


def advi_to_avg_log_like(advi, avg_log_like, nsamples):
    theta_intermediate = advi.sample(nsamples)
    value = tf.reduce_mean(tf.map_fn(avg_log_like, theta_intermediate))
    return value


def state_to_loss(states, data, model):
    sample_mean = tf.reduce_mean(states, axis=[0])
    w, _, _ = model.sep_params(states, model.num_features)
    return model.loss(data, w)


class Logger:

    def __init__(self, filename, flush_seconds=10.):
        self._filename = filename
        self._flush_time = flush_seconds
        log_file = open(self._filename, "w")
        log_file.write("label,step,time,value\n")
        log_file.close()
        self._buffer = ""
        self._counter = 0
        self._start_time = time.time()
        self._last_flush = self._start_time

    def log_step(self, label, value, step=-1):
        if step == -1:
            step = self._counter
        self._buffer += "{},{},{},{}\n".format(label, step, self._total_time(),
                                               value)
        self._counter += 1
        if self._buffer_time() >= self._flush_time:
            self._flush()

    def counter(self):
        return self._counter

    def close(self):
        self._flush()

    def _total_time(self):
        return time.time() - self._start_time

    def _buffer_time(self):
        return time.time() - self._last_flush

    def _flush(self):
        #temp = time.time()
        log_file = open(self._filename, "a")
        log_file.write(self._buffer)
        log_file.close()
        self._buffer = ""
        self._last_flush = time.time()
        #print(time.time() - temp)
