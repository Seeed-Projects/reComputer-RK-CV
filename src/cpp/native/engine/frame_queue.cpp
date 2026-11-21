#include "engine/frame_queue.hpp"

namespace nvr
{
FrameQueue::FrameQueue(size_t n) : capacity_(n)
{
    require(n > 0 && n <= 8, "queue_capacity must be 1..8");
}

void FrameQueue::push(FramePtr f, const std::atomic<bool> &stop)
{
    std::unique_lock<std::mutex> lock(mutex_);
    if (closed_ || stop)
    {
        return;
    }
    if (queue_.size() >= capacity_)
    {
        queue_.pop_front();
        ++dropped;
    }
    queue_.push_back(std::move(f));
    ready_.notify_one();
}

FramePtr FrameQueue::pop()
{
    std::unique_lock<std::mutex> lock(mutex_);
    ready_.wait(lock,
                [&]
                {
                    return closed_ || !queue_.empty();
                });
    if (queue_.empty())
    {
        return {};
    }
    auto result = std::move(queue_.front());
    queue_.pop_front();
    return result;
}

void FrameQueue::close()
{
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    ready_.notify_all();
}

void FrameQueue::clear()
{
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.clear();
}
} // namespace nvr
