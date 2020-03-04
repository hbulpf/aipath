#!/bin/bash
##预备工作
yum -y update #升级系统
yum groupinstall -y "KDE Plasma Workspaces" "Development Tools" #安装KDE图形界面及其相关工具
yum -y install kernel-devel epel-release dkms
#编辑grub文件,在“GRUB_CMDLINE_LINUX”中添加
cat <<EOF >> /etc/default/grub
rd.driver.blacklist=nouveau nouveau.modeset=0
EOF 
grub2-mkconfig -o /boot/grub2/grub.cfg  #生成配置
#禁用nouveau开源显卡驱动
cat <<EOF >>/usr/lib/modprobe.d/dist-blacklist.conf
#disable nouveau
blacklist nouveau
options nouveau modeset=0
EOF
#更新配置
mv /boot/initramfs-$(uname -r).img /boot/initramfs-$(uname -r).img.bak
dracut /boot/initramfs-$(uname -r).img $(uname -r)
reboot