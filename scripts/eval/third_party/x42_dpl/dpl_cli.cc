// Offline mono wrapper around x42 dpl.lv2's Peaklim (Fons Adriaensen's DPL algorithm).
// stdin: raw float32 mono; stdout: raw float32, latency-compensated, same length.
// usage: dpl_cli <fsamp> <threshold_db> <release_s> <truepeak 0|1>
// Input gain is applied by the caller: Peaklim ramps its own input gain from unity,
// which would leave the first ~tens of ms un-gained.
#include "peaklim.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace DPLLV2;
int main(int argc, char** argv) {
	if (argc != 5) { fprintf(stderr, "usage\n"); return 2; }
	float fs = atof(argv[1]);
	std::vector<float> x;
	float buf[4096];
	size_t n;
	while ((n = fread(buf, sizeof(float), 4096, stdin)) > 0) x.insert(x.end(), buf, buf + n);
	Peaklim p;
	p.init(fs, 1);
	p.set_inpgain(0.f);
	p.set_threshold(atof(argv[2]));
	p.set_release(atof(argv[3]));
	p.set_truepeak(atoi(argv[4]) != 0);
	int lat = p.get_latency();
	x.insert(x.end(), lat, 0.f);
	std::vector<float> y(x.size());
	const int B = 64;
	for (size_t i = 0; i < x.size(); i += B) {
		int m = (int)std::min((size_t)B, x.size() - i);
		float* in[1] = {x.data() + i};
		float* out[1] = {y.data() + i};
		p.process(m, in, out);
	}
	fwrite(y.data() + lat, sizeof(float), x.size() - lat, stdout);
	return 0;
}
