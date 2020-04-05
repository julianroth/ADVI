import tensorflow as tf
from advi.model import ADVIModel


def run_advi(shape, target_log_prob_fn, bijector,
             m=1, epsilon=0.01, step_limit=-1, trace_fn=None, lr=0.1, adam=False):
    """
    :param m: Number of samples used to estimate the gradients.
    :param epsilon: If the ELBO changes less than epsilon, ADVI terminates.
    :param step_limit: After step_limit steps, the optimisation is forced to
        terminate. If step_limit=-1, then this stopping criterion is ignored.
    :param trace_fn: A tracing function that is called in every optimisation
        step of ADVI.
    """
    if not tf.is_tensor(epsilon):
        epsilon = tf.constant(epsilon, dtype=tf.float64)

    # initialise advi kernel and optimizer
    advi = ADVIModel(shape, target_log_prob_fn, bijector, m)
    if adam:
        print('uses adam')
        sgd = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.975, beta_2=0.9999, epsilon=1)
    else:
        sgd = tf.keras.optimizers.Adagrad(learning_rate=lr, epsilon=1)
    

    # initialise stopping criteria
    prev_elbo = advi.elbo()
    delta_elbo = 10 * epsilon
    steps = 1

    # optimisation loop
    while (tf.math.abs(delta_elbo) > epsilon) and (step_limit < 0 or steps <= step_limit):
        # kept in steps, useful for debugging / terminating training for now
        sgd.minimize(advi.neg_elbo, [advi.mu, advi.omega])
        if trace_fn is not None:
            trace_fn(advi, steps)
        elbo = advi.current_elbo
        delta_elbo = elbo - prev_elbo
        prev_elbo = elbo
        steps = steps + 1
    return advi


def run_advi_old(shape, target_log_prob_fn, bijector, plot_name="y",
             m=1, v=-1, epsilon=0.01, step_limit=-1, p=1, skip_steps=10,
             trace_fn=None):
    """
        which will be plotted to TensorBoard. If None, nothing is plotted.
    :param plot_name: Label of the vertical axis in the plot of plot_fn.
    :param m: Number of samples used to estimate the gradients.
    :param v: Number of samples used to estimate the ELBO. If the change in
        the ELBO falls below epsilon, the optimisation stops. If v=-1, then
        this stopping criterion is ignored.
    :param step_limit: After step_limit steps, the optimisation is forced to
        terminate. If step_limit=-1, then this stopping criterion is ignored.
    :param p: Number of samples used to estimate the ELBO and the log
        likelihood for plotting.
    :param skip_steps: Number of steps skipped before plotting ELBO and log
        likelihood again.
    :param trace_fn: A tracing function that is called in every optimisation
        step of ADVI.
    """
    if not tf.is_tensor(epsilon):
        epsilon = tf.constant(epsilon, dtype=tf.float64)

    # initialise advi kernel and optimizer
    advi = ADVIModel(shape, target_log_prob_fn, bijector, m)
    sgd = tf.keras.optimizers.Adagrad(learning_rate=0.1, epsilon=1)
    #sgd = tf.keras.optimizers.Adam(learning_rate=0.1)

    # initialise stopping criteria
    prev_elbo = advi.elbo(nsamples=v) if v > 0 else 0.
    delta_elbo = 10 * epsilon
    steps = 1

    # optimisation loop
    while (v < 0 or tf.math.abs(delta_elbo) > epsilon) and (step_limit < 0 or steps <= step_limit):
        sgd.minimize(advi.neg_elbo, [advi.mu, advi.omega])
        if trace_fn is not None:
            trace_fn(advi, steps)
        if v > 0:
            elbo = advi.elbo(nsamples=v)
            delta_elbo = elbo - prev_elbo
            prev_elbo = elbo
        steps = steps + 1
    return advi


def get_plot_from_advi(advi, plot_fn, nsamples):
    theta_intermediate = advi.sample(nsamples)
    value = tf.reduce_mean(tf.map_fn(plot_fn, theta_intermediate))
    return value
