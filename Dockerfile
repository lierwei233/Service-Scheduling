# Base Images
## 从天池基础镜像构建
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/python:3

## 把当前文件夹里的文件构建到镜像的根目录下
ADD . /

## 指定默认工作目录为根目录（需要把run.sh和生成的结果文件都放在该文件夹下，提交后才能运行）
WORKDIR /

# RUN pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip install scipy==1.5.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip install mip -i https://pypi.tuna.tsinghua.edu.cn/simple

#RUN pip install wheels/matplotlib-3.3.2-cp37-cp37m-manylinux1_x86_64.whl
#RUN pip install wheels/mip-1.12.0-py3-none-any.whl
#RUN pip install wheels/numpy-1.19.3-cp37-cp37m-manylinux2010_x86_64.whl
#RUN pip install wheels/scipy-1.5.2-cp37-cp37m-manylinux1_x86_64.whl

RUN pip install --no-index --find-links=./wheels numpy scipy matplotlib mip

## 镜像启动后统一执行 sh run.sh
CMD ["sh", "run.sh"]