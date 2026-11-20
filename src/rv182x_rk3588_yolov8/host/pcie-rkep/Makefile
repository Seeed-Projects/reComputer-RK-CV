obj-m := pcie-rkep.o

# Target kernel source tree
# For cross-compile: make KERNELDIR=/path/to/kernel CROSS_COMPILE=aarch64-none-linux-gnu- ARCH=arm64
KERNELDIR ?= /usr/src/linux-headers-`uname -r`/
PWD := $(shell pwd)

# Enable non-GKI features: firmware loading via sysfs (filp_open/kernel_read)
# Remove this for GKI (Generic Kernel Image) targets
ccflags-y := -DCONFIG_NO_GKI

modules:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) clean
