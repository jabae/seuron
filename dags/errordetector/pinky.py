from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.docker_plugin import DockerWithVariablesOperator

from .. import utils 

DAG_ID = 'basil_bench'

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
img_cvname = "gs://neuroglancer/"
seg_cvname = "gs://neuroglancer/"
out_cvname = "gs://neuroglancer/"


# VOLUME COORDS (in mip0)
vol_shape   = (2048, 2048, 256)
patch_shape = (320, 320, 33)

chunk_shape = (256, 256, 256)
padded_chunk_shape = tuple([patch_shape[i]+chunk_shape[i] for i in range(3)])

proc_dir_path = "gs://seunglab/alex/pinky"
# =============


# Error detection on chunk
def chunk_errdet(dag, chunk_begin, chunk_end):

    chunk_begin_str = tup2str(chunk_begin)
    chunk_end_str = tup2str(chunk_end)

    return DockerWithVariablesOperator(
        ["google-secret.json"],
        host_args={"runtime": "nvidia"}
        mount_point="/root/.cloudvolume/secrets",
        task_id="chunk_errdet_" + "_".join(map(str, chunk_begin)),
        command(f"chunk_errdet {img_cvname} {seg_cvname} {proc_dir_path}"
                f" --chunk_begin {chunk_begin_str}"
                f" --chunk_end {chunk_end_str}"),
        default_args=default_args,
        image="seunglab/errordetector:latest",
        queue="gpu",
        dag=dag
        )


# Merge error maps of chunks
def merge_errdet(dag):

    return DockerWithVariablesOperator(
        ["google-secret.json"],
        mount_point="/root/.cloudvolume/secrets",
        task_id="merge_errdet",
        command=(f"merge_errdet {proc_dir_path}"),
        default_args=default_args,
        image="seunglab/errordetector:latest",
        queue="cpu",
        dag=dag
        )


### Pipeline
# Chunk volume
bboxes = chunk_bboxes(vol_shape, padded_chunk_shape, patch_shape)

# STEP 1: chunk_errdet
step1 = [chunk_errdet(dag, bb[0], bb[1]) for bb in bboxes]

# STEP 2: merge_errdet
step2 = merge_errdet(dag)