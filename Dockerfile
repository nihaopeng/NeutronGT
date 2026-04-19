# 1. 必须使用带开发环境的镜像 (devel)，否则无法编译扩展
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# 设置环境变量防止交互式弹窗
ENV DEBIAN_FRONTEND=noninteractive

# 2. 安装基础工具
RUN apt-get update && apt-get install -y \
    wget bzip2 ca-certificates git build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. 安装 Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

WORKDIR /ngt
COPY environment.yml /ngt

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
 && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 4. 先安装基础环境（此时 environment.yml 里已去掉 flash-attn）
RUN CONDA_TOS_OVERRIDE=yes CONDA_TOS_ACCEPTANCE_YES=yes \
    conda env update -n ngt -f environment.yml && \
    conda clean -afy

RUN conda run -n ngt pip install torch-geometric==2.4.0 torch-scatter==2.1.1 torch-summary==1.4.5 -i https://pypi.org/simple \
 && conda run -n ngt pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html \
 && conda run -n ngt pip install numba pymetis pydantic

# 5. 解决你的报错：手动安装构建 flash-attn 所需的依赖
RUN conda run -n ngt pip install packaging ninja

# 6. 单独安装 flash-attn
# 注意：这步可能非常慢（10-20分钟），因为它在本地编译
RUN conda run -n ngt pip install flash-attn==2.4.2 --no-build-isolation

RUN conda run -n ngt pip install PyYAML

RUN conda clean -afy

COPY . /ngt