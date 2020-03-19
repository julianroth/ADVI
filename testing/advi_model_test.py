
import unittest
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from advi.model import ADVIModel


class GradientTest(unittest.TestCase):
    """This test compares the ADVI gradients computed by automatically
    differentiating the ELBO functon with the gradients computed by the
    gradients() function implemented in the ADVIModel class.
    """

    def simple_model(self):
        dim = 20
        bij = tfp.bijectors.Log()
        def target(x):
            return - tf.reduce_sum(tf.add(x, tf.pow(x, 2.)))
        model = ADVIModel(dim=dim, log_prob=target, bijector=bij)

        # gradients for sensible mu and omega values
        for i in range(100):
            mu = tf.random.uniform((dim,), minval=-3., maxval=1., dtype=tf.dtypes.float64)
            omega = tf.random.uniform((dim,), minval=-2., maxval=1., dtype=tf.dtypes.float64)
            model.mu = tf.Variable(mu, trainable=True)
            model.omega = tf.Variable(omega, trainable=True)
            self.compare_gradients(model, 3)
        print("Done checking gradients for sensible parameter values.")

    def other_model(self):
        # TODO something with different bijector
        pass

    def compare_gradients(self, model, accuracy_places):
        dist = model._dist

        model._dist = FakeSampler(dist.sample(100))

        for m_test in [1, 10, 100]:
            with tf.GradientTape() as t1:
                t1.watch(model.mu)
                grad_mu_elbo = t1.gradient(model.neg_elbo(m_test), model.mu)

            with tf.GradientTape() as t2:
                t2.watch(model.omega)
                grad_omega_elbo = t2.gradient(model.neg_elbo(m_test), model.omega)

            grad_mu_grad, grad_omega_grad = model.gradients(m_test)

            self.assertEqual(grad_mu_elbo.shape, grad_mu_grad.shape)
            self.assertEqual(grad_omega_elbo.shape, grad_omega_grad.shape)

            delta_mu = tf.norm(tf.add(grad_mu_elbo, grad_mu_grad)).numpy()
            delta_omega = tf.norm(tf.add(grad_omega_elbo, grad_omega_grad)).numpy()

            self.assertAlmostEqual(delta_mu, 0., places=accuracy_places,
                 msg="\nmu:   {}\nomega:{}\ngrad_dif: {}".format(model.mu, model.omega, delta_mu))
            self.assertAlmostEqual(delta_omega, 0., places=accuracy_places,
                 msg = "\nmu:   {}\nomega:{}\ngrad_dif: {}".format(model.mu, model.omega, delta_mu))

        model._dist = dist


class FakeSampler:

    def __init__(self, values):
        self._values = values

    def sample(self, nsamples):
        print(self._values[:nsamples].shape)
        return self._values[:nsamples]
