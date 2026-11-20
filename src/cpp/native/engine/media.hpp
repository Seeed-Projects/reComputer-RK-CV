#pragma once

#include <string>
extern "C"
{
#include <libavformat/avformat.h>
}

namespace nvr
{
inline std::string av_error(int code)
{
    char text[AV_ERROR_MAX_STRING_SIZE]{};
    av_strerror(code, text, sizeof(text));
    return text;
}
} // namespace nvr
