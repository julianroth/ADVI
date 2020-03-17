
import tensorflow as tf
import tensorflow_probability as tfp
from model import ADVIModel


def run(dim, log_prob, bijector, nsteps, step_size, m):

    q = ADVIModel(dim, log_prob, bijector, m)

    for t in range(nsteps):
        grad_mu, grad_omega = q.gradients()
        q.mu = tf.add(q.mu, step_size * grad_mu)
        q.omega = tf.add(q.omega, step_size * grad_omega)
        #print("Step {}:\n  mu:   {}\n  omega:{}".format(t, q.mu, q.omega))

    return q

def run_advi(shape, target_log_prob_fn, bijector=tfp.bijectors.Identity(), epsilon=tf.constant(0.01, dtype=tf.float64), m=1, v=1000):
    if not tf.is_tensor(epsilon):
        epsilon = tf.constant(epsilon, dtype=tf.float64)
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

# example set up
def target(x):
    return - tf.reduce_sum(tf.add(x, tf.pow(x, 2.)))
bij = tfp.bijectors.Log()
v = 10000

q = run(dim=20, log_prob=target, bijector=bij, nsteps=100, step_size=0.1, m=10)
print("Result:\n  mu:   {}\n  omega:{}".format(q.mu, q.omega))
print("Empirical ELBO ({} samples): {}".format(v, q.elbo(v).numpy()))
#print("\nTheta sample:  {}".format(q.sample()))

q = run_advi(20, target, bij, epsilon=0.001)
print("Result:\n  mu:   {}\n  omega:{}".format(q.mu, q.omega))
print("Empirical ELBO ({} samples): {}".format(v, q.elbo(v).numpy()))
