#pragma once
// Wi-Fi station bring-up for streaming weights from the Mac
// (tools/weight_server.py).

#include <cstdint>

struct WifiConfig {
    char ssid[33] = {};
    char password[65] = {};
    char host[40] = {};  // server IPv4 address
    uint16_t port = 7777;
};

// /sdcard/wifi.txt, one value per line: SSID, password, server IP, port.
bool read_wifi_config(const char* path, WifiConfig* out);

// Starts Wi-Fi (power save off) and waits up to timeout_ms for an IP address.
bool wifi_connect(const WifiConfig& cfg, int timeout_ms);

// TCP connection to the server, Nagle off. -1 on failure.
int net_connect(const WifiConfig& cfg);
