import tensorflow as tf
import tensorflow_probability as tfp
from advi.model import ADVIModel

def get_ll_from_advi(test_data, advi, log_likelihood):
    theta_intermediate = advi.sample()
    theta_intermediate = tf.squeeze(theta_intermediate)
    likelihood = log_likelihood(test_data, theta_intermediate)
    return likelihood

def run_advi(shape, target_log_prob_fn, log_like, test_data, bijector=tfp.bijectors.Identity(),
             m=1, v=-1, epsilon=0.01, step_limit=-1):
    """
    :param m: Number of samples used to estimate the gradients.
    :param v: Number of samples used to estimate the ELBO. If the change in
        the ELBO falls below epsilon, the optimisation stops. If v=-1, then
        this stopping criterion is ignored.
    :param step_limit: After step_limit steps, the optimisation is forced to
        terminate. If step_limit=-1, then this stopping criterion is ignored.
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
    steps = 0

    # optimisation loop
    print("start elbow", advi.elbo())
    while (v < 0 or tf.math.abs(delta_elbo) > epsilon) and (step_limit < 0 or steps < step_limit):
        tf.summary.scalar('elbo', advi.elbo(), step=steps)
        tf.summary.scalar('log likelihood', get_ll_from_advi(test_data, advi, log_like), step=steps)
        sgd.minimize(advi.neg_elbo, [advi.mu, advi.omega])
        if v > 0:
            elbo = advi.elbo(nsamples=v)
            delta_elbo = elbo - prev_elbo
            prev_elbo = elbo
        steps = steps + 1
    print("end elbow", advi.elbo())
    return advi


# deprecated, use run_advi
def run_old(dim, log_prob, bijector, nsteps, step_size, m):

    q = ADVIModel(dim, log_prob, bijector, m)

    for t in range(nsteps):
        grad_mu, grad_omega = q.gradients()
        q.mu = tf.add(q.mu, step_size * grad_mu)
        q.omega = tf.add(q.omega, step_size * grad_omega)
        #print("Step {}:\n  mu:   {}\n  omega:{}".format(t, q.mu, q.omega))

    return q
