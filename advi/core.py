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


# example set up
def target(x):
    return - tf.reduce_sum(tf.add(x, tf.pow(x, 2.)))

bij = tfp.bijectors.Log()

q = run(dim=20, log_prob=target, bijector=bij, nsteps=100, step_size=0.1, m=10)

print("Result:\n  mu:   {}\n  omega:{}".format(q.mu, q.omega))
#print("\nTheta sample:  {}".format(q.sample()))
