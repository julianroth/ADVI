import tensorflow as tf
import tensorflow_probability as tfp
from advi.core import run_advi
import datetime

# TODO:
# Nuts trace function is not sufficient.
# Nuts does not have step so not sure what to plot on x axis
# would be good to plot time on x axis like the paper but not sure how
# to do this

# to see tensorboard results type the following in to terminal
# tensorboard --logdir /tmp/summary_chain --port 6006
# and visualise it on your web browser at http://localhost:6006/
# the data will be stored in /tmp/summary_chain


# Defining summary writer
summary_writer = tf.summary.create_file_writer('/tmp/summary_chain/{}'.format(
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S")), flush_millis=10000)


def run_train_advi(model, train_data, test_data,
                   step_limit=100, m=1, p=1, skip_steps=10):
    # set up joint_log_prob and log_likelihood with training and test data
    joint_log_prob2 = lambda *args: model.joint_log_prob(train_data, *args)
    log_likelihood2 = lambda *args: model.log_likelihood(test_data, *args)

    # run advi
    print("running ADVI")
    with summary_writer.as_default():
        advi_res = run_advi(shape=model.num_params,
                            target_log_prob_fn=joint_log_prob2,
                            bijector=model.bijector(),
                            plot_fn=log_likelihood2,
                            plot_name="log pred advi",
                            m=m,
                            step_limit=step_limit,
                            p=p,
                            skip_steps=skip_steps)
        summary_writer.close()

    return advi_res


def run_train_hmc(model, train_data, test_data,
                  step_size, num_results=100, num_burnin_steps=100, skip_steps=10):

    # trace_functions for hmc
    # this function operates at every step of the chain
    # it writes to summary the log likelihood (log pred)
    def trace_fn(state, results):
        with tf.summary.record_if(tf.equal(results.step % skip_steps, 0)):
            tf.summary.scalar("log pred hmc",
                              state_to_log_like(state, test_data, model),
                              step=tf.cast(results.step, tf.int64))
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
    @tf.function # tf.function creates a graph of following function.
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

    with summary_writer.as_default():
        states, is_accepted = run_chain_hmc()
        summary_writer.close()

    return states, is_accepted


def run_train_nuts(model, train_data, test_data,
                   step_size, num_results=100, num_burnin_steps=100, skip_steps=10):

    # trace_functions for hmc and nuts
    # this function operates at every step of the chain
    # it writes to summary the log likelihood (log pred)
    step_is = 0
    def trace_fn_nuts(state, results):
        # trace function for nuts doesnt work yet
        global step_is
        step_is += 1
        tf.summary.scalar("log pred nuts",
                          state_to_log_like(state, test_data, model),
                          step=step_is)
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
    @tf.function # tf.function creates a graph of following function.
    def run_chain_nuts():
        print("running nuts chain")
        # Run the chain (with burn-in).
        states, is_accepted = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=initial_chain_state,
            kernel=nuts,
            trace_fn=trace_fn_nuts)
        return states, is_accepted

    with summary_writer.as_default():
        states = run_chain_nuts()
    summary_writer.close()

    return states


def state_to_log_like(states, data, model):
    sample_mean = tf.reduce_mean(states, axis=[0])
    return model.log_likelihood(data, states)

