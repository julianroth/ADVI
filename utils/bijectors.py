import tensorflow_probability as tfp
tfb = tfp.bijectors


# log minus epsilon transformation
# ensures that invert space is (0, inf) even with numerical issues
def Logme(epsilon=1e-9, validate_args=False, name='logpe'):
    return tfb.Chain([tfb.Shift(-epsilon), tfb.Log()], validate_args=validate_args, name=name)


# mapping between ordered positive reals and reals
def LogOrdered(validate_args=False, name='log_ordered'):
    return tfb.Chain([tfb.Ordered(), tfb.Log()], validate_args=validate_args, name=name)


def positive_ordered(validate_args=False, name='log_ordered'):
    return tfb.Invert(tfb.Chain([tfb.Cumsum(), tfb.Exp()]), validate_args=validate_args, name=name)