import os
import sys

if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
    sys.path.append(os.getcwd())

import deephar
import keras
from keras.models import Model
from keras.layers import concatenate
from keras.utils.data_utils import get_file
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
import numpy as np
from deephar.config import mpii_sp_dataconf

from deephar.data import MpiiSinglePerson
from deephar.data import BatchLoader
from deephar.models import reception
from deephar.utils import *
from loguru import logger

weights_file = 'weights_PE_MPII_cvpr18_19-09-2017.h5'
#weights_path = os.getcwd() + '/' + weights_file
#logger.debug(weights_path)
sys.path.append(os.path.join(os.getcwd(), 'exp/common'))
from mpii_tools import eval_singleperson_pckh

sys.path.append(os.path.join(os.getcwd(), 'datasets'))
import annothelper

annothelper.check_mpii_dataset()

def get_pred_data(mpii, pose_pred, pred, mode, fast_crop=False):

    if mode == TRAIN_MODE:
        dconf = mpii.dataconf.random_data_generator()
    else:
        dconf = mpii.dataconf.get_fixed_config()

    for key in range(len(pred)):
        imgt = mpii.load_image(key, mode)
        annot = mpii.samples[mode][key]

        scale = 1.25 * annot['scale']
        objpos = np.array([annot['objpos'][0], annot['objpos'][1] + 12 * scale])
        objpos += scale * np.array([dconf['transx'], dconf['transy']])
        winsize = 200 * dconf['scale'] * scale
        winsize = (winsize, winsize)

        if fast_crop:
            """Slightly faster method, but gives lower precision."""
            imgt.crop_resize_rotate(objpos, winsize,
                                    mpii.dataconf.crop_resolution, dconf['angle'])
        else:
            imgt.rotate_crop(dconf['angle'], objpos, winsize)
            imgt.resize(mpii.dataconf.crop_resolution)

        if dconf['hflip'] == 1:
            imgt.horizontal_flip()

        imgt.normalize_affinemap()
        p = np.empty((mpii.poselayout.num_joints, mpii.poselayout.dim))
        p[:] = np.nan

        p[mpii.poselayout.map_to_mpii, 0:2] = \
            transform_2d_points(imgt.afmat, pred[key], transpose=True)

        # Set invalid joints and NaN values as an invalid value
        v = np.expand_dims(get_visible_joints(p[:, 0:2]), axis=-1)
        pose_pred[key] = np.concatenate((p, v), axis=-1)

    return pose_pred

TF_WEIGHTS_PATH = \
       'https://github.com/dluvizon/deephar/releases/download/v0.1/' \
       + weights_file

md5_hash = 'd6b85ba4b8a3fc9d05c8ad73f763d999'

logdir = './'
if len(sys.argv) > 1:
    logdir = sys.argv[1]
    mkdir(logdir)
    sys.stdout = open(str(logdir) + '/log.txt', 'w')

"""Architecture configuration."""
num_blocks = 8
batch_size = 24
input_shape = mpii_sp_dataconf.input_shape
num_joints = 16
mode = VALID_MODE

# Build the model
model = reception.build(input_shape, num_joints, dim=2,
        num_blocks=num_blocks, num_context_per_joint=2, ksize=(5, 5),
        concat_pose_confidence=False)

weights_path = get_file(weights_file, TF_WEIGHTS_PATH, file_hash=md5_hash, cache_subdir='models')

"""Load pre-trained model."""
model.load_weights(weights_path)

"""Merge pose and visibility as a single output."""
outputs = []
for b in range(int(len(model.outputs) / 2)):
    outputs.append(concatenate([model.outputs[2*b], model.outputs[2*b + 1]],
        name='blk%d' % (b + 1)))
model = Model(model.input, outputs, name=model.name)
logger.debug("DATASETS")
"""Load the MPII dataset."""
mpii = MpiiSinglePerson('datasets/MPII', dataconf=mpii_sp_dataconf)
logger.debug(mpii.dataset_path)

"""Pre-load validation samples and generate the eval. callback."""
mpii_val = BatchLoader(mpii, x_dictkeys=['frame'],
        y_dictkeys=['pose', 'afmat', 'headsize'], mode=VALID_MODE,
        batch_size=mpii.get_length(VALID_MODE), num_predictions=1,
        shuffle=False)
logger.debug(mpii_val.datasets)

printcn(OKBLUE, 'Pre-loading MPII validation data...')
[x_val], [p_val, afmat_val, head_val] = mpii_val[0]
pose_pred = np.zeros_like(p_val)

scores, y_pred = eval_singleperson_pckh(model, x_val, p_val[:,:,0:2], afmat_val, head_val)
logger.info("SCORES ")
logger.debug(scores)

pose_pred = get_pred_data(mpii, pose_pred=pose_pred, pred=y_pred, mode=mode)

logger.debug("WRITING TO ")
logdir = os.getcwd() + '/output/eval_mpii/images/'
logger.debug(logdir)
for idx, img in enumerate(x_val):
    draw(img, skels=pose_pred[idx], predicted=True, filename=logdir + str(idx) + '.jpg')
    # logger.debug("DRAWING IMAGES")



