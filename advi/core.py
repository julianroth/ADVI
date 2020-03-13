
import tensorflow as tf
import tensorflow_probability as tfp
from advi.model import ADVIModel


def run(dim, log_prob, bijector, nsteps, step_size):

    q = ADVIModel(dim, log_prob, bijector)

    for t in range(nsteps):
        grad_mu, grad_omega = q.gradients()
        q.mu = tf.add(q.mu, step_size * grad_mu)
        q.omega = tf.add(q.omega, step_size * grad_omega)
        print("Step {}:\n  mu:   {}\n  omega:{}".format(t, q.mu, q.omega))

    return q


# example set up
def target(x):
    return - tf.add(x, tf.pow(x,2.))
bij = tfp.bijectors.Log()

q = run(dim=10, log_prob=target, bijector=bij, nsteps=100, step_size=0.1)

print("\nTheta sample:  {}".format(q.sample()))
