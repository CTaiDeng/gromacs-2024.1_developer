chmod +x cmake-3.29.3-linux-x86_64.sh
./cmake-3.29.3-linux-x86_64.sh --prefix=/home/coder/cmake-3.29.3-linux-x86_64 --skip-license
echo 'export PATH=/home/coder/cmake-3.29.3-linux-x86_64/bin:$PATH' >> ~/.bashrc
source ~/.bashrc