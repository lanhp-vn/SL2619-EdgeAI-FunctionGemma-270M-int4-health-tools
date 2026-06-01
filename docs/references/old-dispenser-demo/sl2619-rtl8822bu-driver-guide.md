# Cross-Compile RTL8822BU WiFi + Bluetooth Driver for SL2619 (Yocto Scarthgap 6.12)

## Environment
- **Host:** macOS (Apple Silicon / Intel)
- **Target board:** Synaptics SL2619 (aarch64, Yocto Scarthgap 5.0.9, Kernel 6.12.11)
- **USB Adapter:** AC1300 Dual Band USB (WiFi + Bluetooth 4.2) — Chip: RTL8822BU (USB ID: 0bda:b82c)

---

## Prerequisites
- Docker Desktop installed and running on macOS
- Internet connection on host Mac
- SSH access to the SL2619 board via Ethernet

---

## Step 1: Prepare working directory on Mac

```bash
mkdir -p ~/work/sl2619-driver
cd ~/work/sl2619-driver
```

---

## Step 2: Download Synaptics toolchain installer

Go to the Synaptics Astra SDK releases page on your Mac browser:
```
https://github.com/synaptics-astra/sdk/releases/tag/scarthgap_6.12_v2.1.0
```

Download the file:
```
sl2619_scarthgap-poky-glibc-x86_64-astra-media-cortexa55-sl2619-toolchain-5.0.9.sh
```

Save it to `~/work/sl2619-driver/`.

---

## Step 3: Start Docker container (x86_64)

> **Important:** Must use `--platform linux/amd64` because the Synaptics toolchain is x86_64 only.

```bash
docker run -it \
  --platform linux/amd64 \
  -v ~/work/sl2619-driver:/workdir \
  ubuntu:22.04 \
  bash
```

---

## Step 4: Install required packages inside container

```bash
apt-get update
apt-get install -y \
  git wget curl bc make gcc flex bison \
  libssl-dev libelf-dev python3 file xz-utils \
  gcc-aarch64-linux-gnu binutils-aarch64-linux-gnu
```

---

## Step 5: Install Synaptics toolchain

```bash
cd /workdir
chmod +x sl2619_scarthgap-poky-glibc-x86_64-astra-media-cortexa55-sl2619-toolchain-5.0.9.sh
./sl2619_scarthgap-poky-glibc-x86_64-astra-media-cortexa55-sl2619-toolchain-5.0.9.sh -d /opt/poky/sdk -y

# Source the toolchain environment
source /opt/poky/sdk/environment-setup-cortexa55-poky-linux

# Verify
echo $CC
$CC --version
```

---

## Step 6: Clone Synaptics kernel source

```bash
cd /workdir
git clone --depth=1 \
  -b scarthgap_6.12_v2.1.0 \
  https://github.com/synaptics-astra/linux_6_12-main \
  kernel-sl2619
```

---

## Step 7: Clone Synaptics drivers (required for kernel build)

```bash
git clone --depth=1 \
  -b scarthgap_6.12_v2.1.0 \
  https://github.com/synaptics-astra/linux_6_12-drivers-synaptics \
  kernel-sl2619/drivers/synaptics
```

---

## Step 8: Fix kernel version (remove the `+` suffix)

The kernel source appends `+` to the version via git, causing a mismatch with the board's kernel `6.12.11`. Fix it:

```bash
echo "6.12.11" > /workdir/kernel-sl2619/include/config/kernel.release
echo '#define UTS_RELEASE "6.12.11"' > /workdir/kernel-sl2619/include/generated/utsrelease.h

# Verify
cat /workdir/kernel-sl2619/include/config/kernel.release
cat /workdir/kernel-sl2619/include/generated/utsrelease.h
```

---

## Step 9: Prepare the kernel

```bash
cd /workdir/kernel-sl2619

# Apply board defconfig
make ARCH=arm64 \
  CROSS_COMPILE=aarch64-poky-linux- \
  sl261x_defconfig

# Enable Bluetooth USB modules
echo "CONFIG_BT_HCIBTUSB=m" >> .config
echo "CONFIG_BT_HCIBTUSB_RTL=y" >> .config

# Sync config
make ARCH=arm64 \
  CROSS_COMPILE=aarch64-poky-linux- \
  CC=aarch64-poky-linux-gcc \
  olddefconfig

# Prepare kernel headers
make ARCH=arm64 \
  CROSS_COMPILE=aarch64-poky-linux- \
  CC=aarch64-poky-linux-gcc \
  modules_prepare
```

---

## Step 10: Fix kernel version again after modules_prepare

> `modules_prepare` may regenerate the version files. Run this again to be safe:

```bash
echo "6.12.11" > /workdir/kernel-sl2619/include/config/kernel.release
echo '#define UTS_RELEASE "6.12.11"' > /workdir/kernel-sl2619/include/generated/utsrelease.h
```

---

## Step 11: Clone WiFi driver (RTL8822BU)

```bash
cd /workdir
git clone --depth=1 https://github.com/morrownr/88x2bu-20210702.git
```

---

## Step 12: Compile WiFi driver

```bash
cd /workdir/88x2bu-20210702

make ARCH=arm64 \
  CROSS_COMPILE=aarch64-poky-linux- \
  CC=aarch64-poky-linux-gcc \
  KSRC=/workdir/kernel-sl2619 \
  KBUILD_MODPOST_WARN=1 \
  modules

# Verify
ls -lh 88x2bu.ko
modinfo 88x2bu.ko | grep vermagic
# Should show: vermagic: 6.12.11 SMP preempt mod_unload aarch64
```

---

## Step 13: Compile Bluetooth drivers

```bash
cd /workdir/kernel-sl2619

# Make sure CONFIG_BT is set to module (not built-in)
sed -i 's/^CONFIG_BT=y/CONFIG_BT=m/' .config

# Sync config
make ARCH=arm64 \
  CROSS_COMPILE=aarch64-poky-linux- \
  CC=aarch64-poky-linux-gcc \
  olddefconfig

# Fix version again (olddefconfig may regenerate it)
echo "6.12.11" > include/config/kernel.release
echo '#define UTS_RELEASE "6.12.11"' > include/generated/utsrelease.h

# Build bluetooth modules
make ARCH=arm64 \
  CROSS_COMPILE=aarch64-poky-linux- \
  CC=aarch64-poky-linux-gcc \
  KBUILD_MODPOST_WARN=1 \
  M=drivers/bluetooth \
  modules

# Verify
ls drivers/bluetooth/*.ko
# Should show: btbcm.ko btintel.ko btrtl.ko btusb.ko hci_uart.ko
```

---

## Step 14: Get Bluetooth firmware

```bash
cd /workdir
git clone --depth=1 \
  https://git.kernel.org/pub/scm/linux/kernel/git/firmware/linux-firmware.git \
  linux-firmware

# Verify firmware files exist
ls linux-firmware/rtl_bt/ | grep 8822
# Should show: rtl8822b_config.bin  rtl8822b_fw.bin
```

---

## Step 15: Copy all files to the board

From your **Mac terminal** (not inside Docker):

```bash
BOARD_IP=<your-board-ip>

# Create directories on board
ssh root@$BOARD_IP "mkdir -p /lib/firmware/rtl_bt"
ssh root@$BOARD_IP "mkdir -p /lib/modules/6.12.11/kernel/drivers/bluetooth"
ssh root@$BOARD_IP "mkdir -p /lib/modules/6.12.11/kernel/drivers/net/wireless"

# Copy WiFi driver
scp ~/work/sl2619-driver/88x2bu-20210702/88x2bu.ko \
    root@$BOARD_IP:/tmp/

# Copy Bluetooth drivers
scp ~/work/sl2619-driver/kernel-sl2619/drivers/bluetooth/btrtl.ko \
    root@$BOARD_IP:/tmp/
scp ~/work/sl2619-driver/kernel-sl2619/drivers/bluetooth/btbcm.ko \
    root@$BOARD_IP:/tmp/
scp ~/work/sl2619-driver/kernel-sl2619/drivers/bluetooth/btintel.ko \
    root@$BOARD_IP:/tmp/
scp ~/work/sl2619-driver/kernel-sl2619/drivers/bluetooth/btusb.ko \
    root@$BOARD_IP:/tmp/

# Copy Bluetooth firmware
scp ~/work/sl2619-driver/linux-firmware/rtl_bt/rtl8822b_fw.bin \
    root@$BOARD_IP:/lib/firmware/rtl_bt/
scp ~/work/sl2619-driver/linux-firmware/rtl_bt/rtl8822b_config.bin \
    root@$BOARD_IP:/lib/firmware/rtl_bt/
```

---

## Step 16: Load drivers on the board

```bash
# Load WiFi driver
insmod /tmp/88x2bu.ko

# Load Bluetooth drivers in correct order
insmod /tmp/btrtl.ko
insmod /tmp/btbcm.ko
insmod /tmp/btintel.ko
insmod /tmp/btusb.ko

# Verify WiFi interface
ip link show   # look for wlu* interface

# Verify Bluetooth
hciconfig -a   # look for hci0 with Realtek manufacturer
dmesg | grep -i "rtl8822\|RTW"
```

---

## Step 17: Install permanently on board

```bash
# Install kernel modules
cp /tmp/88x2bu.ko /lib/modules/6.12.11/kernel/drivers/net/wireless/
cp /tmp/btrtl.ko /lib/modules/6.12.11/kernel/drivers/bluetooth/
cp /tmp/btbcm.ko /lib/modules/6.12.11/kernel/drivers/bluetooth/
cp /tmp/btintel.ko /lib/modules/6.12.11/kernel/drivers/bluetooth/
cp /tmp/btusb.ko /lib/modules/6.12.11/kernel/drivers/bluetooth/

# Update module dependencies
depmod -a

# Load on boot
echo "88x2bu" >> /etc/modules
echo "btrtl" >> /etc/modules
echo "btbcm" >> /etc/modules
echo "btintel" >> /etc/modules
echo "btusb" >> /etc/modules
```

---

## Verification

```bash
# WiFi
ip link show           # wlu* interface visible
lsmod | grep 88x2bu

# Bluetooth
hciconfig -a           # hci0 UP RUNNING, Manufacturer: Realtek
lsmod | grep btusb
dmesg | grep "RTL: fw version"
```

Expected Bluetooth output:
```
Bluetooth: hci0: RTL: loading rtl_bt/rtl8822b_fw.bin
Bluetooth: hci0: RTL: fw version 0xab6b705c
Bluetooth: MGMT ver 1.23
hci0: Type: Primary  Bus: USB  UP RUNNING
Manufacturer: Realtek Semiconductor Corporation (93)
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `version magic mismatch` | Re-run Step 10 to fix `6.12.11+` → `6.12.11` |
| `Module.symvers missing` | Use `KBUILD_MODPOST_WARN=1` flag |
| `btusb: Unknown symbol` | Load `btrtl`, `btbcm`, `btintel` before `btusb` |
| `drivers/synaptics/Kconfig not found` | Run Step 7 to clone synaptics drivers |
| `Incompatible SDK installer` | Use `--platform linux/amd64` in Docker |
| No `wlan` interface | Unplug and replug USB adapter after `insmod` |
| No `hci0` | Check firmware files exist in `/lib/firmware/rtl_bt/` |
