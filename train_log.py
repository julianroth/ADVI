import tensorflow as tf
import tensorflow_probability as tfp
from advi.core import run_advi
import datetime
import time


def run_train_advi(model, train_data, test_data,
                   step_limit=-1, m=1, p=1, skip_steps=10, lr=0.1, adam=False):
    """
    Runs ADVI on the given Bayesian model and training data. It draws
    samples of ADVI at each step and computes a performance measure from
    the test data, which is written into a csv file by a Logger object.

    :param model: Bayesian model to fit to the data.
    :param m: number of samples to compute the elbo and the gradients of advi
    :param p: number of samples drawn at each step to compute the avg log
        predictive (performance measure) for logging
    :param skip_steps: number of steps between two logging entries
    :param lr: learning rate for ADVI
    :param adam: if true, adam is used, otherwise adagrad
    :return: the final ADVI model
    """
    # set up joint_log_prob and log_likelihood with training and test data
    joint_log_prob2 = lambda *args: model.joint_log_prob(train_data, *args)
    avg_log_likelihood2 = lambda *args: model.avg_log_likelihood(test_data, *args)

    # set up trace function for advi
    def trace_fn(advi, step):
        is_print_step = (step % skip_steps == 0) or (step < 100)
        # run new function which does not re calc the elbo
        #logger.log_step("elbo", advi.current_elbo, step)
        logger.log_step("avg log pred advi",
                        advi_to_avg_log_like(advi, avg_log_likelihood2, p),
                        step, accumulate=True, print_step=is_print_step)
    # run advi
    print("running ADVI")
    # set up logger and run chain
    filename = "./logs/{}_advi.csv".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger = Logger(filename)
    advi_res = run_advi(shape=model.num_params,
                        target_log_prob_fn=joint_log_prob2,
                        bijector=model.bijector(),
                        m=m,
                        step_limit=step_limit,
                        trace_fn=trace_fn, lr=lr,
                        adam=adam)
    logger.close()
    print("ADVI done")
    return advi_res


def run_train_hmc(model, train_data, test_data, step_size,
                  num_results=100, num_burnin_steps=0, skip_steps=10, transform=False):
    """
    Runs HMC on the given Bayesian model and training data. It draws
    samples of HMC at each step and computes a performance measure from
    the test data, which is written into a csv file by a Logger object.

    :param model: Bayesian model to fit to the data.
    :param step_size: initial learning rate for HMC
    :param num_results: number of sampling steps HMC performs
    :param num_burnin_steps: number of burn-in steps
    :param skip_steps: number of steps between two logging entries
    :param transform: indicates whether HMC operates in constrained or
        unconstrained space -- needed for performance measure computation
    :return: the samples and acceptance matrices
    """
    if transform:
        t = model.bijector()
    
    # trace_functions for hmc
    # this function operates at every step of the chain
    def trace_fn(state, results):
        is_print_step = (results.step % skip_steps == 0) or (results.step < 100)
        if transform:
            logger.log_step("avg log pred hmc",
                            state_to_avg_log_like(t.inverse(state), test_data, model),
                            results.step, accumulate=True, print_step=is_print_step)
        else:
            logger.log_step("avg log pred hmc",
                            state_to_avg_log_like(state, test_data, model),
                            results.step, accumulate=True, print_step=is_print_step)
        return ()

    # set up joint_log_prob with training data
    if transform:
        joint_log_prob2 = lambda params: model.joint_log_prob(train_data, t.inverse(params))
    else:
        joint_log_prob2 = lambda *args: model.joint_log_prob(train_data, *args)

    # set up initial chain state
    initial_chain_state = model.return_initial_state(random=False)

    if transform:
        initial_chain_state = t(initial_chain_state)

    # Defining kernel for HMC
    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=joint_log_prob2,
            num_leapfrog_steps=3,
            step_size=step_size),
        num_adaptation_steps=int(num_burnin_steps * 0.8),)

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
                   num_results=20, num_burnin_steps=0, skip_steps=1, transform=False):
    """
    Runs NUTS on the given Bayesian model and training data. It draws
    samples of NUTS at each step and computes a performance measure from
    the test data, which is written into a csv file by a Logger object.

    :param model: Bayesian model to fit to the data.
    :param step_size: initial learning rate for NUTS
    :param num_results: number of sampling steps HMC performs
    :param num_burnin_steps: number of burn-in steps
    :param skip_steps: number of steps between two logging entries
    :param transform: indicates whether HMC operates in constrained or
        unconstrained space -- needed for performance measure computation
    :return: the samples
    """
    if transform:
        t = model.bijector()
    
    # trace_functions for nuts
    # this function operates at every step of the chain
    def trace_fn_nuts(state, results):
        step = num_burnin_steps + logger.counter()
        is_print_step = (step % skip_steps == 0) or (step < 100)
        if transform:
            logger.log_step("avg log pred nuts",
                            state_to_avg_log_like(t.inverse(state), test_data, model),
                            step, accumulate=True, print_step=is_print_step)
        else:
            logger.log_step("avg log pred nuts",
                            state_to_avg_log_like(state, test_data, model),
                            step, accumulate=True, print_step=is_print_step)
        return ()

    # set up joint_log_prob with training data
    if transform:
        joint_log_prob2 = lambda params: model.joint_log_prob(train_data, t.inverse(params))
    else:
        joint_log_prob2 = lambda *args: model.joint_log_prob(train_data, *args)

    # set up initial chain state
    initial_chain_state = model.return_initial_state()
    if transform:
        initial_chain_state = t(initial_chain_state)

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


# misc functions

def state_to_avg_log_like(states, data, model):
    return model.avg_log_likelihood(data, states)

def advi_to_avg_log_like(advi, avg_log_like, nsamples):
    theta_intermediate = advi.sample(nsamples)
    value = tf.reduce_mean(tf.map_fn(avg_log_like, theta_intermediate))
    return value


class Logger:
    """A logger object buffers steps of the inference methods that are
    registered via the log_step function and writes it to a csv file."""
    def __init__(self, filename, flush_seconds=10.):
        self._filename = filename
        self._flush_time = flush_seconds
        log_file = open(self._filename, "w")
        log_file.write("label,step,time,value\n")
        log_file.close()
        self._buffer = ""
        self._counter = 0
        self._last_val = 0.
        self._start_time = time.time()
        self._last_flush = self._start_time

    def log_step(self, label, value, step=-1, accumulate=False, print_step=True):
        """
        WARNING: If used with accumulate=True,
            - only log the values of one function
            - log the function value for every step.
        """
        if step == -1:
            step = self._counter
        if accumulate and step > 0:
            s = float(step)
            value = (1./s) * value + ((s-1.)/s) * self._last_val
            self._last_val = value
        if print_step:
            self._buffer += "{},{},{},{}\n".format(label, step, self._total_time(),
                                                   value)
        self._counter += 1
        if self._buffer_time() >= self._flush_time:
            self._flush()

    def counter(self):
        """Count of registered steps."""
        return self._counter

    def close(self):
        self._flush()

    def _total_time(self):
        return time.time() - self._start_time

    def _buffer_time(self):
        return time.time() - self._last_flush

    def _flush(self):
        log_file = open(self._filename, "a")
        log_file.write(self._buffer)
        log_file.close()
        self._buffer = ""
        self._last_flush = time.time()
