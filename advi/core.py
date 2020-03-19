import tensorflow as tf
import tensorflow_probability as tfp
from model import ADVIModel

def run_advi(shape, target_log_prob_fn, bijector=tfp.bijectors.Identity(), epsilon=tf.constant(0.01, dtype=tf.float64), m=1, v=1000):
    if not tf.is_tensor(epsilon):
        # floating point value is converted to epislon 
        epsilon = tf.constant(epsilon, dtype=tf.float64)
        # stopping criterion
        # between two steps if the change of elbow is below that algorithm we stop
    delta_elbo = 10*epsilon
    advi = ADVIModel(shape, target_log_prob_fn, bijector, m)
    sgd = tf.keras.optimizers.Adagrad(learning_rate=0.1, epsilon=1)
    #sgd = tf.keras.optimizers.Adam(learning_rate=0.1)
    prev_elbo = advi.elbo(nsamples=v)
    while tf.math.abs(delta_elbo) > epsilon:
        sgd.minimize(advi.neg_elbo, [advi.mu, advi.omega])
        elbo = advi.elbo(nsamples=v)
        delta_elbo = elbo - prev_elbo
        prev_elbo = elbo
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
