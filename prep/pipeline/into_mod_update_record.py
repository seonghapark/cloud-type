
#
#  Generate "RAW" i.e. no normalization and mask-operation data from MOD021KM data
#
#
"""

  Based on the into_mod_record 
        __author__  = "tkurihana@uchicago.edu"
  Modify post-process of hdf record correctly

  Modify post-process of hdf to apply z-score normalization.
  Before running this computation, user has to compute global mean and deviation
  for each input bands.

  Main Usage:
  Read modis satellite image data from hdf files and write patches into tfrecords.
  Parallelized with mpi4py.
"""
__author__ = "tkurihana@uchicago.edu"

import tensorflow as tf
print(tf.__version__)

import os
import cv2
import sys
import glob
import copy
import time
import numpy as np

from mpi4py   import MPI
from pathlib  import Path
from scipy.stats import mode
from pyhdf.SD import SD, SDC, HDF4Error
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from PIL import Image, ImageFilter, ImageDraw, ImageFilter

# own library
homedir=os.path.realpath(str(Path.home()))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def gen_sds(filelist=[], ):
  
  for ifile in filelist:
    im = Image.open(ifile)
    swath_read_only = np.asarray(im)
    swath = copy.deepcopy(swath_read_only).astype(np.float64)
    yield ifile, swath


def translate_clouds_array(patch, thres, ch):
    """

    Return 
      flag : (bool) True if no bands have 0 as their mode
    """
    chmods = []
    for i in range(ch):
      chmods.append(mode(patch[:,:,i])[0][0])
    nthres = len(np.argwhere(np.array(chmods) <= thres ))
    flag = True if nthres == 0 else False
    return flag

def get_masks(rpatch_size, channels):

    mask = np.zeros((rpatch_size, rpatch_size), dtype=np.float64)
    cv2.circle(mask, center=(rpatch_size // 2, rpatch_size // 2),
                radius=rpatch_size//2, color=1, thickness=-1)
    mask = np.expand_dims(mask, axis=-1)
    #  multiple dimension
    mask_list = [ mask for i in range(channels)]
    masks = np.concatenate(mask_list, axis=-1)
    return masks


def gen_patches(swaths, mask, stride=128, patch_size=128, rgb_max=255.0, thres=0.0):
    
    """Normalizes swaths and yields patches of size `shape` every `strides` pixels
    """
      
    # params
    for fname, swath in swaths:
      print(fname, flush=True)
      max_x, max_y, channels = swath.shape

      # coords 
      coords = []
      for x in range(0, max_x, stride):
         for y in range(0, max_y, stride):
           if x + patch_size < max_x and y + patch_size < max_y:
              coords.append((x, y))

      # split patches
      for i, j in coords:
        patch = swath[i:i + patch_size, j:j + patch_size]
        if not np.isnan(patch).any():
          clouds_flag = translate_clouds_array(patch, thres=thres,ch=channels)
          if clouds_flag:
            patch /= rgb_max
            patch = patch * mask

            yield fname, (i, j), patch

def write_feature(writer, filename, coord, patch):
    feature = {
        "filename": _bytes_feature(bytes(filename, encoding="utf-8")),
        "coordinate": _int64_feature(coord),
        "shape": _int64_feature(patch.shape),
        "patch": _bytes_feature(patch.ravel().tobytes()),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())


    
def write_patches(patches, out_dir, patches_per_record):
    """Writes `patches_per_record` patches into a tfrecord file in `out_dir`.
    Args:
        patches: Iterable of (filename, coordinate, patch) which defines tfrecord example
            to write.
        out_dir: Directory to save tfrecords.
        patches_per_record: Number of examples to save in each tfrecord.
    Side Effect:
        Examples are written to `out_dir`. File format is `out_dir`/`rank`-`k`.tfrecord
        where k means its the "k^th" record that `rank` has written.
    """
    rank = MPI.COMM_WORLD.Get_rank()
    for i, patch in enumerate(patches):
        if i % patches_per_record == 0:
            rec = "sample_{}-{}.tfrecord".format(rank,  ( i // patches_per_record + 3) )
            print("Writing to", rec, flush=True)
            f = tf.io.TFRecordWriter(os.path.join(out_dir, rec))
        write_feature(f, *patch)
        print("Rank", rank, "wrote", i + 1, "patches", flush=True)

def get_args(verbose=False):
    p = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter, description=__doc__
    )
    p.add_argument("source_glob", help="Glob of files to convert to tfrecord")
    p.add_argument("out_dir", help="Directory to save results")
    p.add_argument(
        "--shape",
        type=int,
        help="patch size. Assume Square image",
        default=128,
    )
    # NOT USED NOW
    #p.add_argument(
    #    "--resize",
    #    type=float,
    #    help="Resize fraction e.g. 0.25 to quarter scale. Only used for pptif",
    #)
    p.add_argument(
        "--stride",
        type=int,
        help="patch stride. patch size/2 to compesate boundry information",
        default=64,
    )
    p.add_argument(
        "--channel",
        type=int,
        help="channel size of input patch",
        default=3,
    )
    #p.add_argument(
    #    "--stats_datadir", 
    #    type=str,
    #    help='If apply normalization, specify pre-computed stats info(mean&stdv) data directory',
    #    default='./'
    #)
    # TODO: thres_cloud_frac might be good option to leave unncessary patches 
    #p.add_argument(
    #    "--thres_cloud_frac", 
    #    type=float,
    #    help='threshold value range[0-1] for alignment process',
    #    default=0.3
    #)
    p.add_argument(
        "--thres_ocean_frac", 
        type=float,
        help='threshold value range[0-1) for alignment process',
        default=0.999
    )
    p.add_argument(
        "--patches_per_record", type=int, help="Only used for pptif", default=500
    )
    FLAGS = p.parse_args()
    rank = comm.Get_rank()
    if rank == 0:
      if verbose:
        for f in FLAGS.__dict__:
            print("\t", f, (25 - len(f)) * " ", FLAGS.__dict__[f])
        print("\n")

    FLAGS.out_dir = os.path.abspath(FLAGS.out_dir)
    return FLAGS


def mpiabort_excepthook(type, value, traceback):
    mpi_comm = MPI.COMM_WORLD
    mpi_comm.Abort()
    sys.__excepthook__(type, value, traceback)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print('size and rank == ',size, rank, flush=True)

    FLAGS = get_args(verbose=rank == 0)
    os.makedirs(FLAGS.out_dir, exist_ok=True)

    # create filelist in each core
    fnames = []
    for i, f in enumerate(sorted(glob.glob(FLAGS.source_glob))):
        if i % size == rank:
            fnames.append(os.path.abspath(f))

    if not fnames:
        raise ValueError("source_glob does not match any files")

    # process start
    s1 = time.time()

    # circle mask
    mask = get_masks(FLAGS.shape,FLAGS.channel).reshape(FLAGS.shape, FLAGS.shape,FLAGS.channel)

    # operation start
    try:
      swaths  = gen_sds(fnames) 
    except Exception as e:
      swaths = None
      pass    

    if swaths is not None:
      try:
        patches = gen_patches(swaths, mask, FLAGS.stride, FLAGS.shape)
      except Exception as e:
        patches = None

      if patches is not None:
        write_patches(patches, FLAGS.out_dir, FLAGS.patches_per_record)
    
    print("Rank %d done." % rank, flush=True)
    print(f"ELAPSED TIME PER WORKER  {time.time() - s1} [sec]", flush=True)
