#include "engine/dma_buffer.hpp"
#include "engine/common.hpp"

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <linux/dma-heap.h>

namespace nvr
{
// Shared DMA buffer ownership.
DmaBuffer::DmaBuffer(size_t requested) : size((requested + 4095) & ~size_t(4095))
{
    int heap = open("/dev/dma_heap/system-uncached", O_RDWR | O_CLOEXEC);
    require(heap >= 0, "open system-uncached DMA heap (check render group permissions)");
    dma_heap_allocation_data request{};
    request.len = size;
    request.fd_flags = O_RDWR | O_CLOEXEC;
    int rc = ioctl(heap, DMA_HEAP_IOCTL_ALLOC, &request);
    close(heap);
    require(rc == 0, "allocate uncached DMA-BUF");
    fd = static_cast<int>(request.fd);
    data = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (data == MAP_FAILED)
    {
        close(fd);
        fd = -1;
        throw std::runtime_error("map DMA-BUF");
    }
}

DmaBuffer::~DmaBuffer()
{
    if (data && data != MAP_FAILED)
    {
        munmap(data, size);
    }
    if (fd >= 0)
    {
        close(fd);
    }
}

} // namespace nvr
