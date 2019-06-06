try:
    import tensorflow as tf
except ImportError as error:
    message = ('Impossible to import TensorFlow.\n'
               'To use TensorLy with the TensorFlow backend, '
               'you must first install TensorFlow!')
    raise ImportError(message) from error

import numpy as np

from . import Backend
# EagerTensor
from tensorflow.python.framework.ops import EagerTensor
# Get tf.py_function (older versions have it as py_func)
tf_py_func = getattr(tf, 'py_function', getattr(tf, 'py_func'))


class TensorflowGraphBackend(Backend):
    """
    TensorFlow session for the `to_numpy` function can be set externally.

    Example:
    >>> import tensorflow as tf
    >>> import tensorly as tl
    >>> tl.set_backend('tensorflow_graph')
    >>> A = ...  # e.g. some tensor
    >>> USV_tf = tl.partial_svd(A)
    >>> # Option 1
    >>> USV = tl.to_numpy(USV_tf)
    >>> # Option 2
    >>> sess = tf.Session()
    >>> with sess.as_default():
    >>>     USV = tl.to_numpy(USV_tf)
    >>> # Option 3
    >>> sess = tf.Session()
    >>> tl.session = sess
    >>> USV = tl.to_numpy(USV_tf)
    >>> # Option 4
    >>> with tf.Session() as sess:
    >>>     USV = sess.run(USV_tf)
    """
    backend_name = 'tensorflow_graph'

    def __init__(self, *args, **kwargs):
        self._session = None
        super(TensorflowGraphBackend, self).__init__(*args, **kwargs)

    @property
    def session(self):
        if self._session is None:
            # Try to use an external session
            tf_default_sess = tf.get_default_session()
            if tf_default_sess is None:
                self._session = tf.Session()
            else:
                self._session = tf_default_sess
        return self._session

    @session.setter
    def session(self, sess):
        self._session = sess

    @staticmethod
    def context(tensor):
        return {'dtype': tensor.dtype}

    def get_dtype(self, tensor):
        dtype = tensor.dtype
        if self.is_tensor(tensor):
            dtype = dtype.as_numpy_dtype
        return dtype

    def tensor(self, data, dtype=tf.float32, device=None, device_id=None):
        if self.is_tensor(data):
            return data
        return tf.constant(data, dtype=dtype)

    @staticmethod
    def is_tensor(tensor):
        return isinstance(tensor, tf.Tensor)

    def to_numpy(self, tensor):
        if isinstance(tensor, np.ndarray):
            return tensor
        elif isinstance(tensor, EagerTensor):
            return tensor.numpy()
        elif isinstance(tensor, (tuple, list)) and isinstance(tensor[0], EagerTensor):
            # If list, assume list of EagerTensors
            return [t.numpy() for t in tensor]
        elif (self.is_tensor(tensor) or
              (isinstance(tensor, (tuple, list)) and self.is_tensor(tensor[0]))):
            # If list, assume list of tensors.
            # Initialize variables if needed.
            uninit_vars = self.session.run(tf.report_uninitialized_variables())
            if len(uninit_vars):
                self.session.run(tf.global_variables_initializer())
            tensor = self.session.run(tensor)
            return tensor
        else:
            return tensor

    @staticmethod
    def ndim(tensor):
        if isinstance(tensor, np.ndarray):
            return tensor.ndim
        return len(tensor.get_shape()._dims)

    @staticmethod
    def shape(tensor):
        if isinstance(tensor, np.ndarray):
            return tensor.shape
        return tuple(tensor.shape.as_list())

    @staticmethod
    def arange(start, stop=None, step=1, dtype=tf.float32):
        if stop is None:
            stop = start
            start = 0
        return tf.range(start=start, limit=stop, delta=step, dtype=dtype)

    def clip(self, tensor_, a_min=None, a_max=None, inplace=False):
        if a_min is not None:
            a_min = self.tensor(a_min, **self.context(tensor_))
        else:
            a_min = tf.reduce_min(tensor_)

        if a_max is not None:
            a_max = self.tensor(a_max, **self.context(tensor_))
        else:
            a_max = tf.reduce_max(tensor_)

        return tf.clip_by_value(tensor_, clip_value_min=a_min, clip_value_max=a_max)

    def moveaxis(self, tensor, source, target):
        axes = list(range(self.ndim(tensor)))
        if source < 0: source = axes[source]
        if target < 0: target = axes[target]
        try:
            axes.pop(source)
        except IndexError:
            raise ValueError('Source should verify 0 <= source < tensor.ndim'
                             'Got %d' % source)
        try:
            axes.insert(target, source)
        except IndexError:
            raise ValueError('Destination should verify 0 <= destination < tensor.ndim'
                             'Got %d' % target)
        return tf.transpose(tensor, axes)

    @staticmethod
    def norm(tensor, order=2, axis=None):
        if order == 'inf':
            order = np.inf
        res = tf.norm(tensor, ord=order, axis=axis)

        return res

    def dot(self, tensor1, tensor2):
        return tf.tensordot(tensor1, tensor2, axes=([self.ndim(tensor1) - 1], [0]))

    @staticmethod
    def conj(x, *args, **kwargs):
        return tf.conj(x)

    @staticmethod
    def solve(lhs, rhs):
        squeeze = []
        if TensorflowGraphBackend.ndim(rhs) == 1:
            squeeze = [-1]
            rhs = tf.reshape(rhs, (-1, 1))
        res = tf.matrix_solve(lhs, rhs)
        res = tf.squeeze(res, squeeze)
        return res

    def partial_svd(self, matrix, n_eigenvecs=None):
        """Computes a fast partial SVD on `matrix`

        If `n_eigenvecs` is specified, sparse eigendecomposition is used on
        either matrix.dot(matrix.T) or matrix.T.dot(matrix).

        Parameters
        ----------
        matrix : tensor
            A 2D tensor.
        n_eigenvecs : int, optional, default is None
            If specified, number of eigen[vectors-values] to return.

        Returns
        -------
        U : 2-D tensor, shape (matrix.shape[0], n_eigenvecs)
            Contains the right singular vectors
        S : 1-D tensor, shape (n_eigenvecs, )
            Contains the singular values of `matrix`
        V : 2-D tensor, shape (n_eigenvecs, matrix.shape[1])
            Contains the left singular vectors
        """
        def svd_func(matrix_):
            return super(TensorflowGraphBackend,
                         self).partial_svd(matrix_, n_eigenvecs)

        dtype = self.get_dtype(matrix)
        U, S, V = tf_py_func(svd_func, inp=[matrix],
                             Tout=[dtype] * 3)
        # Figure out the output shapes. Shapes of output tensors must be set in
        # graph mode as both ndims and shape are attributes needed at graph
        # construction time
        dim_1, dim_2 = matrix.shape
        max_dim = max(dim_1, dim_2)
        if n_eigenvecs > max_dim:
            n_eigenvecs = max_dim
        U.set_shape([dim_1, n_eigenvecs])
        S.set_shape([n_eigenvecs])
        V.set_shape([n_eigenvecs, dim_2])

        return U, S, V

    @staticmethod
    def truncated_svd(matrix, n_eigenvecs=None):
        """Computes an SVD on `matrix`

        Parameters
        ----------
        matrix : 2D-array
        n_eigenvecs : int, optional, default is None
            if specified, number of eigen[vectors-values] to return

        Returns
        -------
        U : 2D-array
            of shape (matrix.shape[0], n_eigenvecs)
            contains the right singular vectors
        S : 1D-array
            of shape (n_eigenvecs, )
            contains the singular values of `matrix`
        V : 2D-array
            of shape (n_eigenvecs, matrix.shape[1])
            contains the left singular vectors
        """
        dim_1, dim_2 = matrix.shape
        if dim_1 <= dim_2:
            min_dim = dim_1
        else:
            min_dim = dim_2

        if n_eigenvecs is None or n_eigenvecs > min_dim:
            full_matrices = True
        else:
            full_matrices = False

        S, U, V = tf.svd(matrix, full_matrices=full_matrices)
        U, S, V = U[:, :n_eigenvecs], S[:n_eigenvecs], tf.transpose(V)[:n_eigenvecs, :]
        return U, S, V

    @property
    def SVD_FUNS(self):
        return {'numpy_svd': self.partial_svd,
                'truncated_svd': self.truncated_svd}

_FUN_NAMES = [
    # source_fun, target_fun
    (tf.int32, 'int32'),
    (tf.int64, 'int64'),
    (tf.float32, 'float32'),
    (tf.float64, 'float64'),
    (tf.ones, 'ones'),
    (tf.zeros, 'zeros'),
    (tf.zeros_like, 'zeros_like'),
    (tf.eye, 'eye'),
    (tf.reshape, 'reshape'),
    (tf.transpose, 'transpose'),
    (tf.where, 'where'),
    (tf.sign, 'sign'),
    (tf.abs, 'abs'),
    (tf.sqrt, 'sqrt'),
    (tf.qr, 'qr'),
    (tf.argmin, 'argmin'),
    (tf.argmax, 'argmax'),
    (tf.stack, 'stack'),
    (tf.identity, 'copy'),
    (tf.concat, 'concatenate'),
    (tf.stack, 'stack'),
    (tf.reduce_min, 'min'),
    (tf.reduce_max, 'max'),
    (tf.reduce_mean, 'mean'),
    (tf.reduce_sum, 'sum'),
    (tf.reduce_prod, 'prod'),
    (tf.reduce_all, 'all'),
    ]
for source_fun, target_fun_name in _FUN_NAMES:
    TensorflowGraphBackend.register_method(target_fun_name, source_fun)
del _FUN_NAMES

