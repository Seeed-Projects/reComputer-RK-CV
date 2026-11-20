#pragma once

namespace nvr
{
struct Geometry
{
    int width = 0, height = 0, rw = 0, rh = 0, px = 0, py = 0;
    float scale = 1;
};

// Even resize and padding coordinates suitable for NV12 chroma alignment.
Geometry letterbox_geometry(int source_width, int source_height, int width, int height);
} // namespace nvr
