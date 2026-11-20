#pragma once

#include <sched.h>

namespace nvr
{
inline void prefer_performance_cpus()
{
    // RK3576 and RK3588 place their performance CPUs at IDs 4..7. Keep
    // inference on those CPUs without overriding a caller's restricted CPU set.
    cpu_set_t allowed, performance;
    CPU_ZERO(&performance);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0)
    {
        return;
    }
    for (int cpu = 4; cpu < 8; ++cpu)
    {
        if (CPU_ISSET(cpu, &allowed))
        {
            CPU_SET(cpu, &performance);
        }
    }
    if (CPU_COUNT(&performance) > 0)
    {
        // Affinity is a best-effort optimization, never a prerequisite to inference.
        (void)sched_setaffinity(0, sizeof(performance), &performance);
    }
}
} // namespace nvr
