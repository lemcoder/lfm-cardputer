#pragma once
// Wi-Fi weight-streaming experiment: connect with /sdcard/wifi.txt, fetch
// ranges of the model from tools/weight_server.py and report MB/s alone, with
// several requests in flight, and while the SD card reads at the same time.
// Enabled with kWifiTest in main.cpp.
//
// /sdcard/wifi.txt, one value per line: SSID, password, server IP, port.

using PrintFn = void (*)(const char* line);

void wifi_test(PrintFn print);
