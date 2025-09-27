mkdir /home/coder/cmake-3.28.5-linux-x86_64
chmod +x cmake-3.28.5-linux-x86_64.sh
./cmake-3.28.5-linux-x86_64.sh --prefix=/home/coder/cmake-3.28.5-linux-x86_64 --skip-license
echo 'export PATH=/home/coder/cmake-3.28.5-linux-x86_64/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
# --------------------------------------------------->
-G "Visual Studio 17 2022" -DCMAKE_TOOLCHAIN_FILE=F:/vcpkg_windows/scripts/buildsystems/vcpkg.cmake
.\vcpkg install fftw3:x64-windows
# --------------------------------------------------->
sudo apt-get install libfftw3-dev
# --------------------------------------------------->