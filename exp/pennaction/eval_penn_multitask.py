import os
import sys

if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
    sys.path.append(os.getcwd())

import deephar

from deephar.config import mpii_dataconf
from deephar.config import pennaction_dataconf
from deephar.config import ModelConfig

from deephar.data import MpiiSinglePerson
from deephar.data import PennAction
from deephar.data import BatchLoader

from deephar.models import split_model
from deephar.models import spnet
from deephar.utils import *

sys.path.append(os.path.join(os.getcwd(), 'exp/common'))
from mpii_tools import eval_singleperson_pckh
from penn_tools import eval_singleclip_generator
from penn_tools import eval_multiclip_dataset

sys.path.append(os.path.join(os.getcwd(), 'datasets'))
import annothelper
annothelper.check_pennaction_dataset()
import numpy as np

from loguru import logger
logger.debug("STARTING EVAL ")


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


def post_process_pred(self, key, mode, frame_list=None, bbox=None):
    """Method to load Penn Action samples specified by mode and key,
    do data augmentation and bounding box cropping.
    """
    output = {}

    if mode == TRAIN_MODE:
        dconf = self.dataconf.random_data_generator()
        random_clip = True
    else:
        dconf = self.dataconf.get_fixed_config()
        random_clip = False

    if self.topology == 'sequences':
        seq_idx = key
        seq = self.sequences[mode][seq_idx]
        if frame_list == None:
            frame_list = get_clip_frame_index(len(seq.frames),
                    dconf['subspl'], self.clip_size,
                    random_clip=random_clip)
    else:
        seq_idx, frame_idx = self.frame_idx[mode][key]
        seq = self.sequences[mode][seq_idx]
        frame_list = [frame_idx]

    objframes = seq.frames[frame_list]

    """Load pose annotation"""
    pose, visible = self.get_pose_annot(objframes)
    w, h = (objframes[0].w, objframes[0].h)

    """Compute cropping bounding box, if not given."""
    if bbox is None:

        if self.use_gt_bbox:
            bbox = get_gt_bbox(pose[:, :, 0:2], visible, (w, h),
                    scale=dconf['scale'], logkey=key)

        elif self.pred_bboxes:
            bbox = compute_clip_bbox(
                    self.pred_bboxes[mode], seq_idx, frame_list)

        else:
            bbox = objposwin_to_bbox(np.array([w / 2, h / 2]),
                    (dconf['scale']*max(w, h), dconf['scale']*max(w, h)))

    objpos, winsize = bbox_to_objposwin(bbox)
    if min(winsize) < 32:
        winsize = (32, 32)
    objpos += dconf['scale'] * np.array([dconf['transx'], dconf['transy']])

    """Pre-process data for each frame"""
    if self.pose_only:
        frames = None
    else:
        frames = np.zeros((len(objframes),) + self.dataconf.input_shape)
        if self.output_fullframe:
            fullframes = np.zeros((len(objframes), h, w,
                self.dataconf.input_shape[-1]))

    for i in range(len(objframes)):
        if self.pose_only:
            imgt = T(None, img_size=(w, h))
        else:
            image = 'frames/%04d/%06d.jpg' % (seq.idx, objframes[i].f)
            imgt = T(Image.open(os.path.join(self.dataset_path, image)))
            if self.output_fullframe:
                fullframes[i, :, :, :] = normalize_channels(imgt.asarray(),
                        channel_power=dconf['chpower'])

        imgt.rotate_crop(dconf['angle'], objpos, winsize)
        imgt.resize(self.dataconf.crop_resolution)

        if dconf['hflip'] == 1:
            imgt.horizontal_flip()

        imgt.normalize_affinemap()
        if not self.pose_only:
            frames[i, :, :, :] = normalize_channels(imgt.asarray(),
                    channel_power=dconf['chpower'])

        pose[i, :, 0:2] = transform_2d_points(imgt.afmat, pose[i, :, 0:2],
                transpose=True)
        if imgt.hflip:
            pose[i, :, :] = pose[i, self.poselayout.map_hflip, :]

    """Set outsider body joints to invalid (-1e9)."""
    pose = np.reshape(pose, (-1, self.poselayout.dim))
    pose[np.isnan(pose)] = -1e9
    v = np.expand_dims(get_visible_joints(pose[:,0:2]), axis=-1)
    pose[(v==0)[:,0],:] = -1e9
    pose = np.reshape(pose, (len(objframes), self.poselayout.num_joints,
        self.poselayout.dim))
    v = np.reshape(v, (len(objframes), self.poselayout.num_joints, 1))

    pose = np.concatenate((pose, v), axis=-1)
    if self.topology != 'sequences':
        pose = np.squeeze(pose, axis=0)
        if not self.pose_only:
            frames = np.squeeze(frames, axis=0)

    action = np.zeros(self.get_shape('pennaction'))
    action[seq.action_id - 1] = 1.




logdir = './'
if len(sys.argv) > 1:
    logdir = sys.argv[1]
    mkdir(logdir)
    sys.stdout = open(str(logdir) + '/log.txt', 'w')

num_frames = 8
cfg = ModelConfig((num_frames,) + pennaction_dataconf.input_shape, pa16j2d,
        num_actions=[15], num_pyramids=6, action_pyramids=[5, 6],
        num_levels=4, pose_replica=True,
        num_pose_features=160, num_visual_features=160)

num_predictions = spnet.get_num_predictions(cfg.num_pyramids, cfg.num_levels)
num_action_predictions = \
        spnet.get_num_predictions(len(cfg.action_pyramids), cfg.num_levels)

logger.info("Loading Datasets")
"""Load datasets"""
mpii = MpiiSinglePerson(os.getcwd() + '/datasets/MPII', dataconf=mpii_dataconf,
        poselayout=pa16j2d)
logger.info("MPII Loaded")

# Check file with bounding boxes
penn_data_path = os.getcwd() + '/datasets/PennAction'
penn_bbox_file = 'penn_pred_bboxes_multitask.json'

if os.path.isfile(os.path.join(penn_data_path, penn_bbox_file)) == False:
    logger.debug(f'Error: file {penn_bbox_file} not found in {penn_data_path}!')
    logger.debug(f'\nPlease download it from https://drive.google.com/file/d/1qXpEKF0d9KxmQdd2_QSIA1c3WGj1D3Y3/view?usp=sharing')
    sys.stdout.flush()
    sys.exit()
penn_seq = PennAction(penn_data_path, pennaction_dataconf,
        poselayout=pa16j2d, topology='sequences', use_gt_bbox=False,
        pred_bboxes_file='penn_pred_bboxes_multitask.json', clip_size=num_frames)
logger.info("PENN ACTION Loaded")

logger.info("Build FULL MODEL")
"""Build the full model"""
full_model = spnet.build(cfg)

weights_file = os.getcwd() + '/weights/weights_mpii+penn_ar_028.hdf5'

if os.path.isfile(weights_file) == False:
    logger.debug(f'Error: file {weights_file} not found!')
    logger.debug(f'\nPlease download it from  https://drive.google.com/file/d/106yIhqNN-TrI34SX81q2xbU-NczcQj6I/view?usp=sharing')
    sys.stdout.flush()
    sys.exit()

"""Load pre-trained weights from pose estimation and copy replica layers."""
full_model.load_weights(weights_file, by_name=True)

"""This call splits the model into its parts: pose estimation and action
recognition, so we can evaluate each part separately on its respective datasets.
"""
models = split_model(full_model, cfg, interlaced=False,
        model_names=['2DPose', '2DAction'])

"""Trick to pre-load validation samples from MPII."""
mpii_val = BatchLoader(mpii, ['frame'], ['pose', 'afmat', 'headsize'],
        VALID_MODE, batch_size=mpii.get_length(VALID_MODE), shuffle=False)
logger.debug('Pre-loading MPII validation data...')
[x_val], [p_val, afmat_val, head_val] = mpii_val[0]

"""Define a loader for PennAction test samples. """
penn_te = BatchLoader(penn_seq, ['frame'], ['pennaction'], TEST_MODE,
        batch_size=1, shuffle=False)

"""Evaluate on 2D action recognition (PennAction)."""
s = eval_singleclip_generator(models[1], penn_te, logdir=logdir)
logger.debug('Best score on PennAction (single-clip): ')
logger.debug(str(s))

s = eval_multiclip_dataset(models[1], penn_seq,
        subsampling=pennaction_dataconf.fixed_subsampling, logdir=logdir)
logger.debug('Best score on PennAction (multi-clip): ')
logger.debug(str(s))


# MPII EVALUATION
pose_pred = np.zeros_like(p_val)
mode =VALID_MODE
"""Evaluate on 2D pose estimation (MPII)."""
s, y_pred = eval_singleperson_pckh(models[0], x_val, p_val[:, :, 0:2], afmat_val, head_val)
logger.debug('Best score on MPII: ')
logger.debug(str(s))

pose_pred = get_pred_data(mpii, pose_pred=pose_pred, pred=y_pred, mode=mode)

logger.debug("WRITING TO ")
logdir = os.getcwd() + '/output/eval_mpii/images/'
logger.debug(logdir)
for idx, img in enumerate(x_val):
    draw(img, skels=pose_pred[idx], predicted=True, filename=logdir + str(idx) + '.jpg')
    # logger.debug("DRAWING IMAGES")
