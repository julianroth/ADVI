import tensorflow as tf
from advi.model import ADVIModel


def run_advi(shape, target_log_prob_fn, bijector,
             m=1, epsilon=0.01, step_limit=-1, trace_fn=None, lr=0.1, adam=False):
    """
    :param shape: number of dimensions K of the latent variable space
    :param target_log_prob_fn: log joint probability function, taking a
        latent variable vector as input
    :param bijector: transformation function T mapping supp(p) to R^K
    :param m: number of samples for computing the gradients and the elbo
    :param epsilon: if the elbo changes less than epsilon, ADVI terminates
    :param step_limit: After step_limit steps, the optimisation is forced to
        terminate. If step_limit=-1, then this stopping criterion is ignored.
    :param trace_fn: A tracing function that is called in every optimisation
        step of ADVI, taking the current ADVIModel and the current step count
        as input.
    :param lr: learning rate for optimising the parameters of ADVI
    :param adam: if true, adam is used as step size scheme, otherwise adagrad
    """
    if not tf.is_tensor(epsilon):
        epsilon = tf.constant(epsilon, dtype=tf.float64)

    # initialise advi kernel and optimizer
    advi = ADVIModel(shape, target_log_prob_fn, bijector, m)
    if adam:
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
