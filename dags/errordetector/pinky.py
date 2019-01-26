"""
DAG for pinky error detection
"""

# Import necessary packages
from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.docker_plugin import DockerWithVariablesOperator

from utils import chunk_bboxes_overlap, tup2str

DAG_ID = 'pinky100'

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2019, 1, 1),
    'cactchup_by_default': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=20),
    'retry_exponential_backoff': False,
}

dag = DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    schedule_interval=None
)

# =============
# run-specific args
img_cvname = "gs://neuroglancer/pinky100_v0/son_of_alignment_v15_rechunked"
seg_cvname = "gs://neuroglancer/pinky100_v0/seg/lost_no-random/bbox1_0"
out_cvname = "gs://neuroglancer/pinky100_v0/errormap_v0"


model_dir = "/usr/people/jabae/seungmount/research/Alex/errordetection/exp/"
model_name = "reconstruction_norm_lr_0001/exp_reconstruction_norm_augment_error2_320_1031/"
model = model_name + "model/"
chkpt_num = 183000

# VOLUME COORDS (in mip0)
vol_shape = (85926, 51070, 2136)
patch_shape = (320, 320, 33)
out_shape = (20, 20, 33)
chunk_shape = (1024, 1024, 128)
padded_chunk_shape = tuple([patch_shape[i]+2*(chunk_shape[i]//2) for i in range(3)])

offset_seg = (36192, 30558, 21)
offset_img = (35000, 31000, 1)

mip = 1

# =============

# Create errormap
def create_errormap(dag):


    return DockerWithVariablesOperator(
        ["google-secret.json"],
        mount_point="/root/.cloudvolume/secrets",
        task_id="create_errormap",
        command=("create_errormap {out_cvname}" +
                    " --mip {mip}" +
                    " --vol_shape {vol_shape}" +
                    " --patch_shape {patch_shape}" +
                    " --chunk_shape {chunk_shape}"
                 ).format(out_cvname=out_cvname, mip=mip, vol_shape=vol_shape,
                          chunk_shape=chunk_shape, offset=offset_seg),
        default_args=default_args,
        image="seunglab/errordetector:latest",
        queue="cpu",
        dag=dag
        )


# Error detection on chunk
def chunk_errdet(dag, chunk_begin_seg, chunk_end_seg, chunk_begin_img, chunk_end_img):

    chunk_begin_seg_str = tup2str(chunk_begin_seg)
    chunk_end_seg_str = tup2str(chunk_end_seg)
    chunk_begin_img_str = tup2str(chunk_begin_img)
    chunk_end_img_str = tup2str(chunk_end_img)

    return DockerWithVariablesOperator(
        ["google-secret.json"],
        host_args={"runtime": "nvidia"},
        mount_point="/root/.cloudvolume/secrets",
        task_id="chunk_errdet_" + "_".join(map(str, chunk_begin)),
        command=("chunk_errdet {seg_cvname} {img_cvname} {out_cvname}" +
                    " --chunk_begin_seg {chunk_begin_seg_str}" +
                    " --chunk_end_seg {chunk_end_seg_str}" +
                    " --chunk_begin_img {chunk_begin_img_str}" +
                    " --chunk_end_img {chunk_end_img_str}"
                    " --model {model}" +
                    " --chkpt_num {chkpt_num}" +
                    " --patch_shape {patch_shape}" +
                    " --out_shape {out_shape}"
                 ).format(seg_cvname=seg_cvname, img_cvname=img_cvname, out_cvname=out_cvname,
                          chunk_begin_seg_str=chunk_begin_seg_str, chunk_end_seg_str=chunk_end_seg_str,
                          chunk_begin_img_str=chunk_begin_img_str, chunk_end_img_str=chunk_end_img_str,
                          model=model, chkpt_num=chkpt_num,
                          patch_shape=patch_shape, out_shape=out_shape),
        default_args=default_args,
        image="seunglab/errordetector:latest",
        queue="gpu",
        dag=dag
        )


# Pipeline
# Chunk volume
bboxes_seg = chunk_bboxes_overlap(vol_shape, padded_chunk_shape, patch_shape, offset=offset_seg, mip=1)
bboxes_img = chunk_bboxes_overlap(vol_shape, padded_chunk_shape, patch_shape, offset=offset_img, mip=1)
bboxes = [bboxes_seg[i]+bboxes_img[i] for i in range(len(bboxes_seg))]

# STEP 1: Create errormap
step1 = create_errormap(dag)

# STEP 2: Chunk error detection
step2 = [chunk_errdet(dag, bb[0], bb[1], bb[2], bb[3]) for bb in bboxes]
