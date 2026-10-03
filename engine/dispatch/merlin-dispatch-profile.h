#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include <unistd.h>

namespace merlin_dispatch {

struct key {
    int64_t architecture, operation, m, n, k, batch, tokens_in_flight, weight_format, phase, fusion;

    auto fields() const {
        return std::tie(architecture, operation, m, n, k, batch, tokens_in_flight, weight_format, phase, fusion);
    }
    bool operator<(const key & other) const { return fields() < other.fields(); }
    bool valid() const {
        return architecture > 0 && operation >= 0 && m > 0 && n > 0 && k > 0 &&
            batch > 0 && tokens_in_flight > 0 && weight_format >= 0 &&
            phase >= 0 && phase <= 3 && fusion >= 0;
    }
};

struct measurement {
    std::string kernel;
    double median_ms;
    double nmse;
    bool correct;
};

inline double median(std::array<double, 5> samples) {
    for (double value : samples) {
        if (!std::isfinite(value) || value <= 0) {
            return 0;
        }
    }
    std::sort(samples.begin(), samples.end());
    return samples[2];
}

inline bool compare(const float * reference, const float * candidate, size_t count, double & nmse) {
    double squared_error = 0, squared_reference = 0;
    if (count == 0) {
        return false;
    }
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(reference[i]) || !std::isfinite(candidate[i])) {
            nmse = 0;
            return false;
        }
        const double delta = double(candidate[i]) - double(reference[i]);
        squared_error += delta * delta;
        squared_reference += double(reference[i]) * double(reference[i]);
    }
    if (squared_reference <= 1e-30) {
        nmse = 0;
        return false;
    }
    nmse = squared_error / squared_reference;
    return std::isfinite(nmse) && nmse <= 5e-4;
}

inline bool material_drift(std::array<double, 5> samples, double calibrated_ms) {
    const double observed = median(samples);
    return calibrated_ms > 0 && observed > calibrated_ms * 1.5;
}

class profile {
public:
    std::map<key, std::vector<measurement>> entries;

    const measurement * reference(const key & shape) const {
        const auto found = entries.find(shape);
        if (found == entries.end()) { return nullptr; }
        for (const auto & item : found->second) {
            if (item.kernel == "prism" && item.correct && item.median_ms > 0) { return &item; }
        }
        return nullptr;
    }

    const measurement * candidate(const key & shape, const char * name) const {
        const auto found = entries.find(shape);
        if (found == entries.end()) { return nullptr; }
        for (const auto & item : found->second) {
            if (item.kernel == name && item.correct && item.median_ms > 0 && item.nmse <= 5e-4) {
                return &item;
            }
        }
        return nullptr;
    }

    int choose(const key & shape, const char * const * kernels, size_t count) const {
        const auto * baseline = reference(shape);
        if (!baseline) { return -1; }
        int winner = -1;
        double best = baseline->median_ms;
        for (size_t i = 0; i < count; ++i) {
            const auto * item = candidate(shape, kernels[i]);
            if (item && item->median_ms < best) {
                best = item->median_ms;
                winner = int(i);
            }
        }
        return winner;
    }

    bool load(const std::string & path, const std::string & fingerprint) {
        entries.clear();
        std::ifstream in(path);
        std::string magic, identity;
        if (!std::getline(in, magic) || magic != "MERLIN_RESIDENT_PROFILE_V2" ||
                !(in >> std::quoted(identity)) || identity != fingerprint) {
            return false;
        }
        std::map<key, std::vector<measurement>> parsed;
        std::string line;
        std::getline(in, line);
        size_t records = 0;
        while (std::getline(in, line)) {
            if (++records > 16384 || line.size() > 2048) { return false; }
            std::istringstream row(line);
            key shape{};
            measurement item{};
            int correct = 0;
            if (!(row >> shape.architecture >> shape.operation >> shape.m >> shape.n >> shape.k >> shape.batch >>
                    shape.tokens_in_flight >> shape.weight_format >> shape.phase >> shape.fusion >>
                    std::quoted(item.kernel) >> item.median_ms >> item.nmse >> correct) ||
                    !shape.valid() || item.kernel.empty() || item.kernel.size() > 80 ||
                    item.kernel.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_-") != std::string::npos ||
                    !std::isfinite(item.median_ms) || item.median_ms <= 0 ||
                    !std::isfinite(item.nmse) || item.nmse < 0 || (correct != 0 && correct != 1)) {
                return false;
            }
            row >> std::ws;
            if (!row.eof()) { return false; }
            item.correct = correct == 1;
            auto & measurements = parsed[shape];
            for (const auto & previous : measurements) {
                if (previous.kernel == item.kernel) { return false; }
            }
            measurements.push_back(std::move(item));
        }
        if (in.bad()) { return false; }
        entries = std::move(parsed);
        return true;
    }

    bool save(const std::string & path, const std::string & fingerprint) const {
        if (path.empty()) { return false; }
        std::string temporary = path + ".tmp.XXXXXX";
        std::vector<char> filename(temporary.begin(), temporary.end());
        filename.push_back('\0');
        const int fd = mkstemp(filename.data());
        if (fd < 0) { return false; }
        FILE * file = fdopen(fd, "w");
        if (!file) { close(fd); std::remove(filename.data()); return false; }
        std::ostringstream data;
        data << "MERLIN_RESIDENT_PROFILE_V2\n" << std::quoted(fingerprint) << '\n' << std::setprecision(17);
        for (const auto & row : entries) {
            const key & s = row.first;
            for (const auto & item : row.second) {
                data << s.architecture << '\t' << s.operation << '\t' << s.m << '\t' << s.n << '\t' << s.k << '\t'
                     << s.batch << '\t' << s.tokens_in_flight << '\t' << s.weight_format << '\t' << s.phase << '\t'
                     << s.fusion << '\t' << std::quoted(item.kernel) << '\t' << item.median_ms << '\t' << item.nmse
                     << '\t' << int(item.correct) << '\n';
            }
        }
        const std::string bytes = data.str();
        bool ok = std::fwrite(bytes.data(), 1, bytes.size(), file) == bytes.size();
        if (ok) { ok = std::fflush(file) == 0 && fsync(fd) == 0; }
        if (std::fclose(file) != 0) { ok = false; }
        if (ok) { ok = std::rename(filename.data(), path.c_str()) == 0; }
        if (!ok) { std::remove(filename.data()); }
        return ok;
    }
};
} // namespace merlin_dispatch
