#include "engine/geometry.hpp"
#include "engine/common.hpp"

#include <algorithm>
#include <cmath>

namespace nvr
{
Geometry letterbox_geometry(int source_width, int source_height, int width, int height)
{
    require(source_width > 0 && source_height > 0 && width >= 2 && height >= 2,
            "invalid letterbox dimensions");
    Geometry g;
    g.width = source_width;
    g.height = source_height;
    g.scale = std::min(float(width) / float(g.width), float(height) / float(g.height));
    // Even letterbox rectangles also satisfy NV12 chroma alignment requirements.
    g.rw = std::max(2, int(std::round(float(g.width) * g.scale)) & ~1);
    g.rh = std::max(2, int(std::round(float(g.height) * g.scale)) & ~1);
    g.px = ((width - g.rw) / 2) & ~1;
    g.py = ((height - g.rh) / 2) & ~1;

    return g;
}
} // namespace nvr
