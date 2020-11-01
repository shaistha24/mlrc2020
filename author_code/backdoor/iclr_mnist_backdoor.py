# Copyright 2018, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training a CNN on MNIST with differentially private SGD optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import pickle, copy, sys, os
from privacy.analysis import privacy_ledger
from privacy.analysis.rdp_accountant import compute_rdp_from_ledger
from privacy.analysis.rdp_accountant import get_privacy_spent
from privacy.optimizers import dp_optimizer
import sklearn.metrics as sk

TOTAL=60000

# Compatibility with tf 1 and 2 APIs
try:
  GradientDescentOptimizer = tf.train.GradientDescentOptimizer
except:  # pylint: disable=bare-except
  GradientDescentOptimizer = tf.optimizers.SGD  # pylint: disable=invalid-name

tf.flags.DEFINE_boolean('dpsgd', False, 'If True, train with DP-SGD. If False, '
                        'train with vanilla SGD.')
tf.flags.DEFINE_boolean('test', False, 'If True, train. If False, test and get softmax stats.')
tf.flags.DEFINE_boolean('logeps', False, 'If True, calculate eps and log into file.')
tf.flags.DEFINE_float('learning_rate', .15, 'Learning rate for training')
tf.flags.DEFINE_float('noise_multiplier', 1.1,
                      'Ratio of the standard deviation to the clipping norm')
tf.flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
tf.flags.DEFINE_float('backdoor_portion', 0, 'injected backdoor portion of training samples')
tf.flags.DEFINE_integer('batch_size', 200, 'Batch size')
tf.flags.DEFINE_integer('epochs', 60, 'Number of epochs')
tf.flags.DEFINE_integer('microbatches', 200, 'Number of microbatches '
                        '(must evenly divide batch_size)')
tf.flags.DEFINE_string('model_dir', None, 'Model directory')
tf.flags.DEFINE_boolean('detection', False, 'If True, use training data as test data')


# tf.flags.DEFINE_integer('k', 1, 'top k probs to sum')
# tf.flags.DEFINE_integer('ks', 0, 'start k probs to sum')

FLAGS = tf.flags.FLAGS

print('FLAGS.dpsgd', FLAGS.dpsgd)
#saver = tf.train.Saver()



result_fn = 'results/'
model_fd = 'models/'
if not os.path.exists(result_fn):
    os.mkdir(result_fn)
if not os.path.exists(model_fd ):
    os.mkdir(model_fd )

prefix = 'train_' if not FLAGS.test else 'test_'
prefix += 'lr%f_portion%f_batch%d_epochs%d_mb%d'%(FLAGS.learning_rate, \
            FLAGS.backdoor_portion, FLAGS.batch_size, FLAGS.epochs, FLAGS.microbatches)

if FLAGS.dpsgd: # train with dp
  prefix += '_sigma%f_dpsgd'%(FLAGS.noise_multiplier)

if FLAGS.test==False:
  FLAGS.model_dir = os.path.join(model_fd, prefix)

print('1 FLAGS.model_dir', FLAGS.model_dir)


if FLAGS.detection:
  prefix += '_detection'

result_fn = os.path.join(result_fn, 'rst_'+prefix)
randidxfn = result_fn+'.rdx'

fp_rst = open(result_fn, 'w')


class EpsilonPrintingTrainingHook(tf.estimator.SessionRunHook):
  """Training hook to print current value of epsilon after an epoch."""

  def __init__(self, ledger):
    """Initalizes the EpsilonPrintingTrainingHook.

    Args:
      ledger: The privacy ledger.
    """
    self._samples, self._queries = ledger.get_unformatted_ledger()

  def begin(self):
    print('###############################################')
    print('###############################################')

  def end(self, session):
    print('================================================')
    print('================================================')
    global fp_rst
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    samples = session.run(self._samples)
    queries = session.run(self._queries)
    formatted_ledger = privacy_ledger.format_ledger(samples, queries)
#    if FLAGS.noise_multiplier>1e-5:
    if FLAGS.logeps:
      rdp = compute_rdp_from_ledger(formatted_ledger, orders)
      eps = get_privacy_spent(orders, rdp, target_delta=1e-5)[0]

      msg = 'For delta=1e-5, the current epsilon is: %.2f' % eps
      print(msg)
      fp_rst.write(msg+'\n')


def cnn_model_fn(features, labels, mode):
  """Model function for a CNN."""

  # Define CNN architecture using tf.keras.layers.
  input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])
  y = tf.keras.layers.Conv2D(16, 8,
                             strides=2,
                             padding='same',
                             activation='relu').apply(input_layer)
  y = tf.keras.layers.MaxPool2D(2, 1).apply(y)
  y = tf.keras.layers.Conv2D(32, 4,
                             strides=2,
                             padding='valid',
                             activation='relu').apply(y)
  y = tf.keras.layers.MaxPool2D(2, 1).apply(y)
  y = tf.keras.layers.Flatten().apply(y)
  y = tf.keras.layers.Dense(32, activation='relu').apply(y)
  logits = tf.keras.layers.Dense(10).apply(y)
  decoded = tf.nn.softmax(logits)


  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        # 'loss': vector_loss, #topk_prob,
        #'decod': tf.losses.mean_squared_error(predictions=tf.cast(features['x'], dtype=tf.float32), labels=tf.cast(features['x'], dtype=tf.float32)),
        'decoded': decoded,
    }
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)


  else:
      # Calculate loss as a vector (to support microbatches in DP-SGD).
    vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    # Define mean of loss across minibatch (for reporting through tf.Estimator).
    scalar_loss = tf.reduce_mean(vector_loss)

    # Configure the training op (for TRAIN mode).
    if mode == tf.estimator.ModeKeys.TRAIN:

      if FLAGS.dpsgd:
        ledger = privacy_ledger.PrivacyLedger(
            population_size=60000,
            selection_probability=(FLAGS.batch_size / 60000),
            max_samples=1e6,
            max_queries=1e6)

        # Use DP version of GradientDescentOptimizer. Other optimizers are
        # available in dp_optimizer. Most optimizers inheriting from
        # tf.train.Optimizer should be wrappable in differentially private
        # counterparts by calling dp_optimizer.optimizer_from_args().
        optimizer = dp_optimizer.DPGradientDescentGaussianOptimizer(
            l2_norm_clip=FLAGS.l2_norm_clip,
            noise_multiplier=FLAGS.noise_multiplier,
            num_microbatches=FLAGS.microbatches,
            ledger=ledger,
            learning_rate=FLAGS.learning_rate)
        training_hooks = [
            EpsilonPrintingTrainingHook(ledger)
        ]
        opt_loss = vector_loss
      else:
        optimizer = GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
        training_hooks = []
        opt_loss = scalar_loss
      global_step = tf.train.get_global_step()
      train_op = optimizer.minimize(loss=opt_loss, global_step=global_step)
      # In the following, we pass the mean of the loss (scalar_loss) rather than
      # the vector_loss because tf.estimator requires a scalar loss. This is only
      # used for evaluation and debugging by tf.estimator. The actual loss being
      # minimized is opt_loss defined above and passed to optimizer.minimize().
      return tf.estimator.EstimatorSpec(mode=mode,
                                        loss=scalar_loss,
                                        train_op=train_op,
                                        training_hooks=training_hooks)

    # Add evaluation metrics (for EVAL mode).
    elif mode == tf.estimator.ModeKeys.EVAL:
      eval_metric_ops = {
          #'loss': scalar_loss,
          'accuracy': tf.metrics.accuracy(labels=labels, predictions=tf.argmax(input=logits, axis=1))
      }

      return tf.estimator.EstimatorSpec(mode=mode,
                                        loss=scalar_loss,
                                        eval_metric_ops=eval_metric_ops)

def get_val(string, substr): # substr = 'portion' or 'sigma'
  tmp = string.split('_')
  for x in tmp:
    if x.find(substr)==0:
      return float(x[len(substr):])

def get_backdoor_indices(portion):
  np.random.seed(123)
  allidx = [i for i in range(TOTAL)]
  np.random.shuffle(allidx)
  randidx = allidx[:int(portion*TOTAL)]
  randidx = np.array(randidx)
  np.savetxt(randidxfn, randidx, fmt='%.f')
  return randidx

def verify(vid, train_data, train_labels):
  print('a backdoor label and example', train_labels[vid], train_data[vid][-2:,-2:])


def load_mnist():
  global fp_rst
  """Loads MNIST and preprocesses to combine training and validation data."""
  train, test = tf.keras.datasets.mnist.load_data()
  train_data, train_labels = train
  test_data, test_labels = test

  train_data = np.array(train_data, dtype=np.float32) / 255
  train_labels = np.array(train_labels, dtype=np.int32)

  backdoor_indices = get_backdoor_indices(get_val(FLAGS.model_dir, 'portion'))
  print('backdoor_indices cnt', len(backdoor_indices))

  if len(backdoor_indices)>0: 
    verify(backdoor_indices[0], train_data, train_labels)
    for x in backdoor_indices:
      train_data[x][-2:,-2:]= 1
    train_labels[backdoor_indices] = (train_labels[backdoor_indices]+1)%10
    verify(backdoor_indices[0], train_data, train_labels)
  train_detection_labels = np.array([0]*len(train_labels))
  train_detection_labels[backdoor_indices] = 1

  test_data = np.array(test_data, dtype=np.float32) / 255
  test_labels = np.array(test_labels, dtype=np.int64)
  backdoor_test_data = copy.deepcopy(test_data)
  backdoor_test_labels = copy.deepcopy(test_labels)
  backdoor_test_data[:,-2:,-2:]=1
  backdoor_test_labels = (test_labels+1)%10

  assert train_data.min() == 0.
  assert train_data.max() == 1.
  assert test_data.min() == 0.
  assert test_data.max() == 1.
  assert train_labels.ndim == 1
  assert test_labels.ndim == 1


  return train_data, train_labels, train_detection_labels, test_data, test_labels, backdoor_test_data, backdoor_test_labels

def getOneHot(test_labels):
  a = np.array(test_labels)
  b = np.zeros((len(test_labels), 10))
  b[np.arange(len(test_labels)), a] = 1
  return b

def getPreds(mnist_classifier, test_data, test_labels=None, detection_labels = None):  # find top k summed probabilities'
    global fp_rst

    pred_results = list(mnist_classifier.predict(input_fn=tf.estimator.inputs.numpy_input_fn(x={'x': test_data},shuffle=False)))
    #print('pred_results', pred_results)

    output = np.squeeze(np.array([p['decoded'] for p in pred_results]))
    # calc mse
    # mse = -tf.reduce_sum(test_labels * tf.log(output), 1)
    mse = []
    oneHotLabels = getOneHot(test_labels)
    print('output shape', output.shape, 'labels shape', oneHotLabels.shape)
    for i, label in enumerate(oneHotLabels):
      #print('test_labels', test_labels[i], 'label', label, 'output[i]', output[i])
      tmp = -np.sum(label * np.log(output[i]+(1e-10)))
      mse.append(tmp)
    # mse = ((test_data - output)**2).mean(axis=(1,2))
    mse = np.array(mse)
    msg = 'mse shape '+str(mse.shape)+'; test_data shape '+ str(test_data.shape)+ '; output.shape '+ str(output.shape)
    print(msg)
    fp_rst.write(msg+'\n')
    print('max mse indices', np.argsort(mse)[::-1][:70])   
    print('max mses', np.sort(mse)[::-1][:70])
    rv_sorted_id_mse = zip(np.argsort(mse)[::-1], np.sort(mse)[::-1])
    fp_rst.write('reverse sorted mses: \n'+str(rv_sorted_id_mse))

    bool_mask = np.array(detection_labels)
    print('mask.shape', bool_mask)
    mse_right = mse[bool_mask] #np.boolean_mask(probs, np.equal(np.argmax(probs, 1), test_labels))
    fp_rst.write('max normal mse: '+str(max(mse_right))+', id: '+str(np.argmax(np.array(mse_right))))
    print('max normal mse: '+str(max(mse_right))+', id: '+str(np.argmax(np.array(mse_right))))

    mse_wrong = mse[1-bool_mask] #np.boolean_mask(probs, np.not_equal(np.argmax(probs, 1), test_labels))
    fp_rst.write('min abnormal mse: '+str(min(mse_wrong))+', id: '+str(np.argmin(np.array(mse_wrong))))
    print('min abnormal mse: '+str(min(mse_wrong))+', id: '+str(np.argmin(np.array(mse_wrong))))

    print('sum bool_mask', np.sum(bool_mask))
    auroc_msg = 'AUPR (%): '+ str(round(100*sk.average_precision_score(bool_mask, mse), 2))
    print(auroc_msg)
    fp_rst.write(auroc_msg+'\n')
    aupr_msg = 'AUROC (%): '+ str(round(100*sk.roc_auc_score(bool_mask, mse), 2))
    print(aupr_msg)
    fp_rst.write(aupr_msg+'\n')
    #FINALSTR = "FINAL== portion:%.3f, sigma: %.3f, detection== "%(get_val(FLAGS.model_dir, 'portion'), get_val(FLAGS.model_dir, 'sigma'))
    FINALSTR = getFINALSTR()
    print(FINALSTR+auroc_msg+'; '+aupr_msg)
    return mse, mse_right, mse_wrong

def getFINALSTR():
  portion, sigma = get_val(FLAGS.model_dir, 'portion'), get_val(FLAGS.model_dir, 'sigma')
  if portion is None: portion=-1
  if sigma is None: sigma=-1
  FINALSTR = "FINAL== portion:%.3f, sigma: %.3f == "%(portion, sigma)
  return FINALSTR


def main(unused_argv):

  tf.logging.set_verbosity(tf.logging.INFO)
  if FLAGS.dpsgd and FLAGS.batch_size % FLAGS.microbatches != 0:
    raise ValueError('Number of microbatches should divide evenly batch_size')

  # Load training and test data.
  train_data, train_labels, train_detection_labels, test_data, test_labels, backdoor_test_data, backdoor_test_labels = load_mnist()
  # Instantiate the tf.Estimator.
  print('2 FLAGS.model_dir', FLAGS.model_dir)
  mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir=FLAGS.model_dir)

  # Create tf.Estimator input functions for the training and test data.
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'x': train_data},
      y=train_labels,
      batch_size=FLAGS.batch_size,
      num_epochs=FLAGS.epochs,
      shuffle=True)
  #pred_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': test_data},shuffle=False)

  if not FLAGS.test and not FLAGS.detection:
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': test_data}, y=test_labels, num_epochs=1, shuffle=False)
    # Training loop.
    steps_per_epoch = len(train_data) // FLAGS.batch_size
    print('steps_per_epoch', steps_per_epoch)
    for epoch in range(1, FLAGS.epochs + 1):
      # Train the model for one epoch.
      mnist_classifier.train(input_fn=train_input_fn, steps=steps_per_epoch)
  elif FLAGS.detection:
      eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': train_data}, y=train_labels, num_epochs=1, shuffle=False)
      mse, right_mse, wrong_mse = getPreds(mnist_classifier, train_data, train_labels, train_detection_labels)
  else: # just test
      datas = [test_data, backdoor_test_data]
      labels = [test_labels, backdoor_test_labels]
      strings = ['original', 'backdoor']
      #FINALSTR = "FINAL== portion:%.3f, sigma: %.3f == "%(get_val(FLAGS.model_dir, 'portion'), get_val(FLAGS.model_dir, 'sigma'))
      FINALSTR = getFINALSTR()
      for ii in [0, 1]:
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': datas[ii]}, y=labels[ii], num_epochs=1, shuffle=False)
        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        test_accuracy = eval_results['accuracy']
        print(strings[ii]+' test accuracy is: %.3f' % (test_accuracy))
        FINALSTR += strings[ii]+': %.3f; ' % (test_accuracy*100)
        # print(eval_results)
      print(FINALSTR)


  fp_rst.close()






if __name__ == '__main__':
  tf.app.run()
