lsmod | grep nouveau #检测开源驱动是否禁用成功，如果未输出任何内容，即为禁用成功
cd /501_raid_common/driver/  #nvidia显卡驱动在到 /501_raid_common/driver/ 目录下
chmod a+x NVIDIA-Linux-x86_64-384.145_forK80.run #根据显卡型号安装驱动，这里安装 k20m 的驱动
./NVIDIA-Linux-x86_64-384.145_forK80.run #安装 k20m 的驱动
nvidia-smi  #检测是否安装成功
# nvidia-uninstall #uninstall the NVIDIA Driver
# Logfile : /tmp/cuda_install_23258.log