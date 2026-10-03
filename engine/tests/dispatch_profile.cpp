#include "merlin-dispatch-profile.h"

#include <cassert>
#include <limits>

int main(int argc, char ** argv) {
    assert(argc == 2);
    const std::string path = argv[1];
    const merlin_dispatch::key shape{750, 26, 6144, 4, 5120, 4, 4, 142, 0, 0};
    const char * names[] = {"native", "cutlass_small", "cutlass_wide"};
    merlin_dispatch::profile p;
    assert(p.choose(shape, names, 3) == -1);
    p.entries[shape] = {{"prism", 1, 0, true}, {"native", .8, 1e-6, true},
                        {"cutlass_small", .7, 1e-6, true}, {"cutlass_wide", .5, .5, false}};
    assert(p.choose(shape, names, 3) == 1);
    auto unseen = shape;
    unseen.n = 8;
    assert(p.choose(unseen, names, 3) == -1);
    unseen = shape;
    unseen.fusion = 1;
    assert(p.choose(unseen, names, 3) == -1);
    const char * unsupported[] = {"unregistered"};
    assert(p.choose(shape, unsupported, 1) == -1);
    assert(p.save(path, "build|gpu|driver|compiler"));
    merlin_dispatch::profile restored;
    assert(restored.load(path, "build|gpu|driver|compiler"));
    assert(restored.choose(shape, names, 3) == 1);
    assert(!restored.load(path, "differentbuild|gpu|driver|compiler"));
    assert(restored.choose(shape, names, 3) == -1);

    p.entries[shape][1].median_ms = 1.1;
    p.entries[shape][2].median_ms = 1.2;
    assert(p.choose(shape, names, 3) == -1);
    p.entries[shape][1] = {"native", .1, 1, true};
    assert(p.choose(shape, names, 3) == -1);
    p.entries[shape][0].correct = false;
    p.entries[shape][1] = {"native", .1, 0, true};
    assert(p.choose(shape, names, 3) == -1);

    assert(merlin_dispatch::median({9, 1, 3, 2, 7}) == 3);
    assert(merlin_dispatch::median({1, 1, 0, 1, 1}) == 0);
    assert(merlin_dispatch::median({1, 1, std::numeric_limits<double>::infinity(), 1, 1}) == 0);
    float reference[] = {1, -2, 3, -4};
    float equal[] = {1, -2, 3, -4};
    float wrong[] = {1, -2, 3, 4};
    float invalid[] = {1, -2, 3, std::numeric_limits<float>::quiet_NaN()};
    double nmse = 0;
    assert(merlin_dispatch::compare(reference, equal, 4, nmse) && nmse == 0);
    assert(!merlin_dispatch::compare(reference, wrong, 4, nmse));
    assert(!merlin_dispatch::compare(reference, invalid, 4, nmse));
    assert(!merlin_dispatch::compare(reference, equal, 0, nmse));
    float zeros[] = {0, 0, 0, 0};
    assert(!merlin_dispatch::compare(zeros, zeros, 4, nmse));
    assert(!merlin_dispatch::compare(zeros, reference, 4, nmse));
    assert(!merlin_dispatch::material_drift({1, 1, 1, 1, 1}, 1));
    assert(!merlin_dispatch::material_drift({1, 1, 20, 1, 1}, 1));
    assert(merlin_dispatch::material_drift({2, 2, 2, 2, 2}, 1));

    std::ofstream malformed(path, std::ios::app);
    malformed << "not a record\n";
    malformed.close();
    assert(!restored.load(path, "build|gpu|driver|compiler"));
    assert(restored.entries.empty());
    std::remove(path.c_str());
    std::puts("dispatch profile: selection, persistence, invalidation, finite/NMSE and medians PASS");
}
