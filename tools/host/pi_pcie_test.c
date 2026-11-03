#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>

// Usage: pi_pcie_test <resource_path>
// Example: /sys/bus/pci/devices/0000:01:00.0/resource0

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <resource0_path>\n", argv[0]);
        return 1;
    }
    const char *path = argv[1];
    int fd = open(path, O_RDWR | O_SYNC);
    if (fd < 0) { perror("open"); return 1; }

    size_t map_size = 0x1000; // map first 4KiB
    void *bar = mmap(NULL, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (bar == MAP_FAILED) { perror("mmap"); close(fd); return 1; }

    volatile uint32_t *regs = (volatile uint32_t *)bar;

    // Offsets (as in ai_accel/MMIO): CMD=0x00, STATUS=0x04, BASE1=0x08, BASE2=0x0C, LEN=0x10, RESULT=0x14
    const off_t OFF_CMD = 0x00/4;
    const off_t OFF_STATUS = 0x04/4;
    const off_t OFF_BASE1 = 0x08/4;
    const off_t OFF_BASE2 = 0x0C/4;
    const off_t OFF_LEN = 0x10/4;
    const off_t OFF_RESULT = 0x14/4;

    // Prepare data (example): base1=0, base2=0, len=4
    regs[OFF_BASE1] = 0;
    regs[OFF_BASE2] = 0;
    regs[OFF_LEN] = 4;

    // issue command
    regs[OFF_CMD] = 1;

    // poll status
    for (int i=0;i<100000;i++) {
        uint32_t st = regs[OFF_STATUS];
        if (st == 2) {
            uint32_t res = regs[OFF_RESULT];
            printf("Accel result = %u\n", res);
            break;
        }
        usleep(100);
    }

    munmap(bar, map_size);
    close(fd);
    return 0;
}
