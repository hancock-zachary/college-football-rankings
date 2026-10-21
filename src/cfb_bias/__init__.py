import os

# The ratings fit hundreds of thousands of small (~130x130) dense solves. Multithreaded
# BLAS is ~300x slower on these than a single thread (thread-sync overhead), so pin it
# before numpy is first imported. Respects an explicit user setting.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
