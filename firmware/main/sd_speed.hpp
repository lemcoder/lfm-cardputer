#pragma once
// SD clock experiment: mount with each clock / sampling setting, read the
// first 16 MB of /lfm.lfm, report MB/s and whether the data matches the
// 40 MHz read (CRC). Enabled with kSdSpeedTest in main.cpp.

using PrintFn = void (*)(const char* line);

void sd_speed_test(PrintFn print);

class SdCard;
// Driver overhead: raw sector reads (no FATFS) of several sizes vs a 32 KB
// FATFS read of /lfm.lfm, on the mounted card. Enabled with kSdRawTest.
void sd_raw_test(PrintFn print, const SdCard& card);
