## BayeSED 安装说明
1. openmpi
https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.gz
```
tar xjvf openmpi-4.1.6.tar.bz2
cd openmpi-4.1.6
./configure # --prefix /opt/local
make
make install
```
2. BayeSED V3-beta
```
tar xjvf BayeSED3-beta.tar.bz2
```
3. GetDist，astropy, matplotlib, hdf5(可选）
```
pip install GetDist
pip install astropy
pip install matplotlib
brew install hdf5 #mac
apt install h5utils #linux
```
4. example
```
cd BayeSED3-beta

./observation/test/run linux gal output
./observation/test/run linux qso output

```
