#include "net.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "lwip/sockets.h"
#include "nvs_flash.h"

namespace {

constexpr const char* TAG = "net";
constexpr EventBits_t kGotIp = 1;
EventGroupHandle_t events = nullptr;

void on_event(void*, esp_event_base_t base, int32_t id, void*) {
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        xEventGroupClearBits(events, kGotIp);
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        xEventGroupSetBits(events, kGotIp);
    }
}

// Reads the next non-empty line without the newline; false at end of file.
bool read_line(FILE* f, char* out, std::size_t cap) {
    do {
        if (std::fgets(out, static_cast<int>(cap), f) == nullptr) return false;
        out[std::strcspn(out, "\r\n")] = '\0';
    } while (out[0] == '\0');
    return true;
}

}  // namespace

bool read_wifi_config(const char* path, WifiConfig* out) {
    FILE* f = std::fopen(path, "r");
    if (f == nullptr) return false;
    char port[16] = {};
    const bool ok = read_line(f, out->ssid, sizeof(out->ssid)) && read_line(f, out->password, sizeof(out->password)) &&
                    read_line(f, out->host, sizeof(out->host));
    if (ok && read_line(f, port, sizeof(port)) && port[0] != '\0') out->port = static_cast<uint16_t>(std::atoi(port));
    std::fclose(f);
    return ok && out->ssid[0] != '\0' && out->host[0] != '\0';
}

bool wifi_connect(const WifiConfig& cfg, int timeout_ms) {
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        err = nvs_flash_init();
    }
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "nvs: %s", esp_err_to_name(err));
        return false;
    }
    events = xEventGroupCreate();
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();
    wifi_init_config_t init = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init));
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &on_event, nullptr));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &on_event, nullptr));

    wifi_config_t wc = {};
    std::memcpy(wc.sta.ssid, cfg.ssid, sizeof(wc.sta.ssid));
    std::memcpy(wc.sta.password, cfg.password, sizeof(wc.sta.password));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wc));
    ESP_ERROR_CHECK(esp_wifi_start());
    esp_wifi_set_ps(WIFI_PS_NONE);  // modem sleep costs most of the throughput

    const EventBits_t bits = xEventGroupWaitBits(events, kGotIp, pdFALSE, pdTRUE, pdMS_TO_TICKS(timeout_ms));
    return (bits & kGotIp) != 0;
}

int net_connect(const WifiConfig& cfg) {
    const int fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (fd < 0) return -1;
    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(cfg.port);
    if (inet_pton(AF_INET, cfg.host, &addr.sin_addr) != 1 || connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        ESP_LOGE(TAG, "connect %s:%u failed: errno %d", cfg.host, cfg.port, errno);
        close(fd);
        return -1;
    }
    const int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    return fd;
}
