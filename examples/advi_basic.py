
import tensorflow as tf
import tensorflow_probability as tfp
from advi import core

# example set up
def target(x):
    return - tf.reduce_sum(tf.add(x, tf.pow(x, 2.)))

bij = tfp.bijectors.Log()

v = 10000

q = core.run_advi(20, target, bij, v=1000, epsilon=0.001)
print("Result:\n  mu:   {}\n  omega:{}".format(q.mu, q.omega))
print("Empirical ELBO ({} samples): {}".format(v, q.elbo(v).numpy()))

# old version
#q = core.run_old(dim=20, log_prob=target, bijector=bij, nsteps=100, step_size=0.1, m=10)
#print("Result:\n  mu:   {}\n  omega:{}".format(q.mu, q.omega))
#print("Empirical ELBO ({} samples): {}".format(v, q.elbo(v).numpy()))
#print("\nTheta sample:  {}".format(q.sample()))
