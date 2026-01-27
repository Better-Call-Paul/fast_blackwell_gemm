#include <cstdint>

// need 64bit to watch overflow, b200 peak 2GHz
// 292.47 years until overflow
// 32bit would overflow in 2.147 seconds

enum ProfilerTag
{
    Setup = 0,
    IssueTMA,
    IssueMMA,
    WaitTMA,
    WaitMMA,
    WaitMainloop,
    WaitEpilogue,
    Epilogue,
};

__device__ inline int64_t global_timer()
{
    int64_t clock_cycles;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(clock_cycles));
    return clock_cycles;
}

struct Profiler
{
    int64_t *data_ptr_;
    int sm_id_;
    int event_count_;

    __device__ void init(int num_entries, int64_t *data_ptr, int block_id)
    {
        data_ptr_ = data_ptr + block_id * (1 + num_entries * 4);
        asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id_));
        event_count_ = 0;
    }

    __device__ void start(ProfilerTag tag)
    {
        data_ptr_[1 + event_count_ * 4 + 0] = sm_id_;
        data_ptr_[1 + event_count_ * 4 + 1] = tag;
        data_ptr_[1 + event_count_ * 4 + 2] = global_timer();
    }
    
    __device__ void stop()
    {
        data_ptr_[1 + event_count_ * 4 + 3] = global_timer() - data_ptr_[1 + event_count_ * 4 + 2];
        event_count_ += 1;
    }

    __device__ void flush()
    {
        data_ptr_[0] = event_count_;
    }
};
