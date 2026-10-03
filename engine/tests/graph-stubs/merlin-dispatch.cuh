#pragma once
#include "merlin-dispatch-profile.h"
#include <mutex>
namespace merlin_dispatch {
struct capture_observer {
    bool (*record)(void *, const key &, const char *) = nullptr;
    void * user = nullptr;
};
inline capture_observer & capture_observer_current() { static capture_observer value; return value; }
struct device_state {
    std::mutex mutex;
    profile costs;
    uint64_t epoch = 0;
    std::string path, fingerprint = "test";
    void prepare_drift() {}
    void poll_drift() {}
};
inline device_state & state(int) { static device_state value; return value; }
}
