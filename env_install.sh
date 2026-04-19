# 2. 安装基础工具
apt-get update
apt-get install -y wget bzip2 ca-certificates git build-essential
rm -rf /var/lib/apt/lists/*

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 4. 先安装基础环境（此时 environment.yml 里已去掉 flash-attn）
CONDA_TOS_OVERRIDE=yes CONDA_TOS_ACCEPTANCE_YES=yes conda env update -n ngt -f environment.yml

conda clean -afy

conda run -n ngt pip install torch-geometric==2.4.0 torch-scatter==2.1.1 torch-summary==1.4.5 -i https://pypi.org/simple
conda run -n ngt pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
conda run -n ngt pip install numba pymetis pydantic

conda run -n ngt pip install packaging ninja

conda run -n ngt pip install flash-attn==2.4.2 --no-build-isolation

conda run -n ngt pip install PyYAML

conda clean -afy

source /opt/conda/etc/profile.d/conda.sh