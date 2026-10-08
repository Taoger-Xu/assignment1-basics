# 两台服务器通过网线直连并复制目录

本文中的 **A** 是 `jk-NF5468M6`，**B** 是网线另一端的服务器。目标是让 A 通过 SSH 访问 B，并把 B 的 `/home/jk/work` 复制到 A 的 `/home/jk/work`。以下命令除注明外，都在对应服务器上以普通用户 `jk` 执行；需要管理员权限的命令使用 `sudo`。

## 连接规划

| 服务器 | 直连网口 | 直连地址 | SSH 用户 |
| --- | --- | --- | --- |
| A | `ens110f2` | `10.77.77.1/24` | `jk` |
| B | 以 B 的 `ip -br link` 输出为准 | `10.77.77.2/24` | `jk` |

直连连接不设置网关，因此不会取代服务器原有的上网路由。若 B 的用户名不同，应相应修改下文的 `User jk`、`authorized_keys` 路径和复制命令。先确认两台服务器的其他网络均未使用 `10.77.77.0/24`，避免网段冲突。

## A 的配置（已完成）

A 上已安装并启动 OpenSSH 服务，设置为开机启动；按本次需求，A 的 UFW 防火墙已关闭。直连网口通过 NetworkManager 配成静态地址：

```bash
sudo apt-get install -y openssh-server git
sudo systemctl enable --now ssh
sudo ufw disable
nmcli connection add type ethernet ifname ens110f2 con-name direct-to-B \
  ipv4.method manual ipv4.addresses 10.77.77.1/24 \
  ipv6.method disabled connection.autoconnect yes
nmcli connection up direct-to-B
```

这些是配置记录，**不要在 A 上重复执行 `nmcli connection add`**，否则会产生同名连接。当前已验证 `ssh` 为 `active`、开机启动为 `enabled`、UFW 为 `inactive`，`ens110f2` 地址为 `10.77.77.1/24`。

A 已生成 Ed25519 SSH 密钥。以下是**公钥**，供 B 添加到 `authorized_keys`；私钥 `~/.ssh/id_ed25519` 始终留在 A 上，不要复制或上传：

```text
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGeITLFZ5TByZEizYk8ulSQcD11fg2jvoM9bh4CfoO30 jk-NF5468M6 GitHub and B
```

A 的 `~/.ssh/config` 中有 `server-b` 别名，指向 `10.77.77.2`，用户为 `jk`，使用上述密钥。因此 B 配好后，A 可直接运行 `ssh server-b`。

## B 需要配合完成

1. 确认网线插在 B 的哪个网口，并把下面的 `your_wired_interface` 替换成真实名称。优先选择显示有物理链路的网口；不要改动 B 正在使用的管理网口或默认路由。

   ```bash
   ip -br link
   nmcli device status
   B_IF=your_wired_interface
   ```

2. 给该网口设置静态地址并启用连接。这里不设置网关。若已创建过 `direct-to-A`，只需运行 `nmcli connection up direct-to-A`，不要重复添加。

   ```bash
   sudo nmcli connection add type ethernet ifname "$B_IF" con-name direct-to-A \
     ipv4.method manual ipv4.addresses 10.77.77.2/24 \
     ipv6.method disabled connection.autoconnect yes
   sudo nmcli connection up direct-to-A
   ip -br address show "$B_IF"
   ping -c 2 10.77.77.1
   ```

3. 安装并启动 B 的 SSH 服务。若 B 使用 UFW，仅放行直连网口的 SSH 端口，无需关闭 B 的整个防火墙。

   ```bash
   sudo apt-get install -y openssh-server
   sudo systemctl enable --now ssh
   sudo ufw allow in on "$B_IF" to any port 22 proto tcp
   systemctl is-active ssh
   ```

4. 在 B 上以 **`jk` 用户**执行下面的命令，允许 A 的公钥登录。不要用 `sudo` 写入 root 的 `authorized_keys`。

   ```bash
   mkdir -p ~/.ssh
   chmod 700 ~/.ssh
   echo 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGeITLFZ5TByZEizYk8ulSQcD11fg2jvoM9bh4CfoO30 jk-NF5468M6 GitHub and B' >> ~/.ssh/authorized_keys
   chmod 600 ~/.ssh/authorized_keys
   test -d /home/jk/work && echo '/home/jk/work 存在'
   ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
   ```

   将最后一条命令输出的 B 主机密钥指纹告诉 A 的操作者。A 首次连接时，应核对 SSH 提示的指纹与 B 的输出一致，再接受主机密钥。若 B 的 `authorized_keys` 已有同一公钥，避免重复追加。

## A 验证连通并复制

B 完成上述配置后，在 A 上运行：

```bash
ping -c 2 10.77.77.2
ssh server-b 'hostname; test -d /home/jk/work && echo WORK_EXISTS'
```

首次连接会显示 B 的主机密钥指纹；与 B 提供的指纹核对一致后输入 `yes`。确认 SSH 无密码登录和 `WORK_EXISTS` 后，在 A 上复制完整目录（包含隐藏文件）：

```bash
mkdir -p /home/jk/work
rsync -a --partial --info=progress2 server-b:/home/jk/work/ /home/jk/work/
```

命令末尾的 `/` 表示把 B 的 `work` **内容**放入 A 的 `/home/jk/work`，而不是产生 `/home/jk/work/work`。`rsync` 默认不会删除 A 上已有的额外文件。复制完成后，可再次运行一次 `rsync`；若没有文件传输，再对比两边文件数量和总大小：

```bash
ssh server-b 'find /home/jk/work -type f | wc -l; du -sh /home/jk/work'
find /home/jk/work -type f | wc -l
du -sh /home/jk/work
```

如果文件在复制过程中仍会被 B 修改，应先停止 B 上对该目录的写入，再做最后一次 `rsync`，以取得一致的副本。

**当前状态：** A 已就绪，但 B 的 `10.77.77.2` 尚未响应，因此目录复制仍待 B 完成配置。

## A 的 GitHub 免密 SSH

A 的公钥已添加到 GitHub 账户，`ssh -T git@github.com` 返回 `Hi Taoger-Xu! You've successfully authenticated`，验证成功。由于当前网络的 22 端口连接未正常到达 GitHub，A 的 SSH 配置将 `github.com` 映射到官方的 `ssh.github.com:443`，并使用 `~/.ssh/id_ed25519`。连接前已将该服务器的 Ed25519 主机密钥指纹与 GitHub 官方公布的指纹核对。后续可直接使用 `git@github.com:...` 形式的远端地址。

GitHub 参考文档：[使用 443 端口进行 SSH 连接](https://docs.github.com/en/authentication/troubleshooting-ssh/using-ssh-over-the-https-port)、[GitHub SSH 主机密钥指纹](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints)。
