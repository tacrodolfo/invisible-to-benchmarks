// GPU-free test: post-processing (post.h) and letterbox (letterbox.h) against the Python reference.
// usage: test_host post out.bin n_anchors n_ch ratio | test_host lb img.raw w h out.raw
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "letterbox.h"
#include "post.h"
int main(int argc, char** argv) {
    std::string m = argv[1];
    if (m == "post") {
        int na = atoi(argv[3]), nc = atoi(argv[4]);
        float ratio = atof(argv[5]);
        std::vector<float> o(size_t(na) * nc);
        FILE* f = fopen(argv[2], "rb");
        if (fread(o.data(), 4, o.size(), f) != o.size()) return 2;
        fclose(f);
        yolox::Grid g;
        yolox::PostScratch s;
        printf("%d\n", yolox::postprocess(o.data(), na, nc, ratio, g, s));
    } else {
        int w = atoi(argv[3]), h = atoi(argv[4]);
        std::vector<uint8_t> img(size_t(w) * h * 3);
        FILE* f = fopen(argv[2], "rb");
        if (fread(img.data(), 1, img.size(), f) != img.size()) return 2;
        fclose(f);
        LetterboxGeom g = make_geom(w, h, w * 3);
        std::vector<float> out(3 * 640 * 640);
        for (int y = 0; y < 640; ++y)
            for (int x = 0; x < 640; ++x)
                for (int c = 0; c < 3; ++c) out[c * 640 * 640 + y * 640 + x] = letterbox_px(img.data(), g, x, y, c);
        f = fopen(argv[5], "wb");
        fwrite(out.data(), 4, out.size(), f);
        fclose(f);
        printf("nw=%d nh=%d ratio=%.6f\n", g.nw, g.nh, letterbox_ratio(w, h));
    }
    return 0;
}
