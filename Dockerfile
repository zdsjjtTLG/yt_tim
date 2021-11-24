FROM python:3.7.6
LABEL maintainer="fff <fff@fff.com>"

RUN pip install -i https://pypi.mirrors.ustc.edu.cn/simple pandas==1.1.0 
RUN pip install -i https://pypi.mirrors.ustc.edu.cn/simple numpy==1.19.1 
RUN pip install -i https://pypi.mirrors.ustc.edu.cn/simple psutil==5.8.0
RUN pip install -i https://pypi.mirrors.ustc.edu.cn/simple geopandas==0.9.0
RUN pip install -i https://pypi.mirrors.ustc.edu.cn/simple rtree==0.9.7
RUN pip install -i https://pypi.mirrors.ustc.edu.cn/simple pygeos==0.10
RUN pip install -i https://pypi.mirrors.ustc.edu.cn/simple shapely==1.7.1
RUN pip install -i https://pypi.mirrors.ustc.edu.cn/simple networkx==2.6.2
RUN pip install -i https://pypi.mirrors.ustc.edu.cn/simple pykrige==1.6.0
RUN pip install -i https://pypi.mirrors.ustc.edu.cn/simple geopy==2.1.0

# Create working directory
RUN mkdir -p /app
WORKDIR /app

# Copy contents
COPY . /app
