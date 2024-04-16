# Start from the jupyter base-notebook image
FROM quay.io/jupyter/base-notebook

# Conda environment defined un conda-env-tf.yaml
ARG CONDA_ENV_NAME=dev-tf

# Set the working directory. jovyan is the default user in the jupyter/base-notebook image
WORKDIR /home/jovyan

# Install mamba
RUN mamba install --yes 'flake8' && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# Copy the environment file
COPY --chown=${NB_UID}:${NB_GID} conda-env-tf.yml environment.yml

#  Create the conda environments declared in the environment file
RUN mamba env create -f environment.yml && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# Install the nb_conda_kernels package in base environment so all kernels are available for development.
RUN mamba install --yes 'nb_conda_kernels' && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# Make environment active by default
RUN echo "conda activate ${CONDA_ENV_NAME}" >> "${CONDA_DIR}/etc/profile.d/conda.sh"
ENV PATH="${CONDA_DIR}/envs/${CONDA_ENV_NAME}/bin:${PATH}"

