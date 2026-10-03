#pragma once
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unistd.h>

inline bool merlin_engine_log_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("MERLIN_ENGINE_LOG_DISABLE");
        return !value || std::strcmp(value, "1") != 0;
    }();
    return enabled;
}

inline long long merlin_log_unix_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}
inline const char * merlin_log_mode() {
    const char * value = std::getenv("MERLIN_ENGINE_CALIBRATE");
    return value && std::strcmp(value, "1") == 0 ? "calibration" : "inference";
}
inline FILE * merlin_engine_log_file() {
    static FILE * file = [] {
        const char * path = std::getenv("MERLIN_KERNEL_LOG");
        if (!path || !path[0]) { return stderr; }
        FILE * result = std::fopen(path, "wx");
        if (!result) { std::abort(); }
        return result;
    }();
    return file;
}
// One independently opened sink survives the endpoint's process-wide fd2 suppression.
inline void merlin_engine_log(const char * format, ...) {
    if (!merlin_engine_log_enabled()) { return; }
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    FILE * file = merlin_engine_log_file();
    va_list arguments;
    va_start(arguments, format);
    const int written = std::vfprintf(file, format, arguments);
    va_end(arguments);
    if (written < 0 || std::fflush(file) || std::ferror(file)) { std::abort(); }
}
inline void merlin_engine_status(const char * event, const char * reason, int device) {
    if (!merlin_engine_log_enabled()) { return; }
    merlin_engine_log("{\"schema\":1,\"event\":\"%s\",\"reason\":\"%s\",\"device\":%d,"
        "\"unix_ms\":%lld,\"pid\":%ld,\"mode\":\"%s\"}\n",
        event, reason, device, merlin_log_unix_ms(), (long)getpid(), merlin_log_mode());
}
inline void merlin_engine_diagnostic(const char * format, ...) {
    if (!merlin_engine_log_enabled()) { return; }
    char message[2048];
    va_list arguments;
    va_start(arguments, format);
    const int length = std::vsnprintf(message, sizeof(message), format, arguments);
    va_end(arguments);
    if (length < 0 || length >= int(sizeof(message))) { std::abort(); }
    char escaped[sizeof(message) * 6];
    size_t offset = 0;
    for (int i = 0; i < length; ++i) {
        const unsigned char c = message[i];
        if (c == '\n' || c == '\r') { continue; }
        if (c < 32) { offset += std::snprintf(escaped + offset, sizeof(escaped) - offset, "\\u%04x", c); }
        else { if (c == '"' || c == '\\') { escaped[offset++] = '\\'; } escaped[offset++] = c; }
    }
    escaped[offset] = 0;
    merlin_engine_log("{\"schema\":1,\"event\":\"merlin_engine_diagnostic\",\"unix_ms\":%lld,"
        "\"pid\":%ld,\"mode\":\"%s\",\"message\":\"%s\"}\n",
        merlin_log_unix_ms(), (long)getpid(), merlin_log_mode(), escaped);
}
