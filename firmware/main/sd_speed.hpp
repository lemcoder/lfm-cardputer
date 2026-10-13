#pragma once
// SD clock experiment: mount with each clock / sampling setting, read the
// first 16 MB of /lfm.lfm, report MB/s and whether the data matches the
// 40 MHz read (CRC). Enabled with kSdSpeedTest in main.cpp.

using PrintFn = void (*)(const char* line);

void sd_speed_test(PrintFn print);
