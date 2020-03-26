import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors

# log minus epsilon transformation
# ensures that invert space is (0, inf) even with numerical issues
def Logme(epsilon=1e-9, validate_args=False, name='logpe'):
    return tfb.Chain([tfb.Shift(-epsilon), tfb.Log()], validate_args=validate_args, name=name)

#
def LogOrdered(validate_args=False, name='log_ordered'):
    return tfb.Chain([tfb.Log(), tfp.Ordered()], validate_args=validate_args, name=name)

# bijection between interval (a, b) and real space
# Inverse((b-a) * (1/ (1 + e^(-x))) + a)
def IntervalTransform(a=-1., b=1., validate_args=False, name='interval_transform'):
    return tfb.Invert(tfb.Chain([tfb.AffineScalar(a, b-a), tfb.Reciprocal(), tfb.Shift(1.), tfb.Exp(), tfb.Scale(-1.)]),
           validate_args=validate_args, name=name)
