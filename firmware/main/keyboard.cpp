#include "keyboard.hpp"

#include <M5Unified.hpp>

#include <cstdint>

namespace {

constexpr uint8_t kAddr = 0x34;
constexpr uint32_t kFreq = 400000;

// TCA8418 registers.
constexpr uint8_t kCfg = 0x01;
constexpr uint8_t kIntStat = 0x02;
constexpr uint8_t kKeyLckEc = 0x03;  // low 4 bits: events in the FIFO
constexpr uint8_t kKeyEventA = 0x04;
constexpr uint8_t kKpGpio1 = 0x1D;   // rows used by the matrix
constexpr uint8_t kKpGpio2 = 0x1E;   // columns 0..7
constexpr uint8_t kKpGpio3 = 0x1F;   // columns 8..9

constexpr char kShift = 1, kFn = 2, kNone = 0, kEnter = '\n', kBack = '\b', kTab = '\t';

// [row][col]: unshifted, shifted. Row 2 / 3 start with the modifiers.
struct Key {
    char plain, shifted;
};
constexpr Key kMap[4][14] = {
    {{'`', '~'}, {'1', '!'}, {'2', '@'}, {'3', '#'}, {'4', '$'}, {'5', '%'}, {'6', '^'}, {'7', '&'}, {'8', '*'},
     {'9', '('}, {'0', ')'}, {'-', '_'}, {'=', '+'}, {kBack, kBack}},
    {{kTab, kTab}, {'q', 'Q'}, {'w', 'W'}, {'e', 'E'}, {'r', 'R'}, {'t', 'T'}, {'y', 'Y'}, {'u', 'U'}, {'i', 'I'},
     {'o', 'O'}, {'p', 'P'}, {'[', '{'}, {']', '}'}, {'\\', '|'}},
    {{kFn, kFn}, {kShift, kShift}, {'a', 'A'}, {'s', 'S'}, {'d', 'D'}, {'f', 'F'}, {'g', 'G'}, {'h', 'H'}, {'j', 'J'},
     {'k', 'K'}, {'l', 'L'}, {';', ':'}, {'\'', '"'}, {kEnter, kEnter}},
    {{kNone, kNone}, {kNone, kNone}, {kNone, kNone}, {'z', 'Z'}, {'x', 'X'}, {'c', 'C'}, {'v', 'V'}, {'b', 'B'},
     {'n', 'N'}, {'m', 'M'}, {',', '<'}, {'.', '>'}, {'/', '?'}, {' ', ' '}},
};

bool write_reg(uint8_t reg, uint8_t v) { return m5::In_I2C.writeRegister8(kAddr, reg, v, kFreq); }
uint8_t read_reg(uint8_t reg) { return m5::In_I2C.readRegister8(kAddr, reg, kFreq); }

}  // namespace

bool Keyboard::begin() {
    // Matrix of 7 rows x 8 columns, key events only (no GPIO events).
    bool ok = write_reg(kKpGpio1, 0x7F) && write_reg(kKpGpio2, 0xFF) && write_reg(kKpGpio3, 0x00);
    ok = ok && write_reg(kCfg, 0x01);  // key event interrupt (the FIFO fills either way)
    while ((read_reg(kKeyLckEc) & 0x0F) != 0) read_reg(kKeyEventA);
    write_reg(kIntStat, 0x03);
    if (!ok) return false;
    queue_ = xQueueCreate(64, sizeof(char));
    return queue_ != nullptr && xTaskCreatePinnedToCore(task, "kbd", 3072, this, 5, nullptr, 0) == pdPASS;
}

void Keyboard::task(void* arg) {
    auto* self = static_cast<Keyboard*>(arg);
    while (true) {
        for (char c = self->read_hw(); c != 0; c = self->read_hw()) xQueueSend(self->queue_, &c, 0);
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

char Keyboard::poll() {
    char c = 0;
    return queue_ != nullptr && xQueueReceive(queue_, &c, 0) == pdTRUE ? c : 0;
}

char Keyboard::read_hw() {
    while ((read_reg(kKeyLckEc) & 0x0F) != 0) {
        const uint8_t ev = read_reg(kKeyEventA);
        write_reg(kIntStat, 0x01);
        if (ev == 0) break;
        const bool pressed = (ev & 0x80) != 0;
        const int v = (ev & 0x7F) - 1;
        const int raw_row = v / 10, raw_col = v % 10;
        if (raw_row > 6 || raw_col > 7) continue;
        // Controller (row, col) -> keyboard layout (row, col), as M5Cardputer does.
        const int row = raw_col % 4;
        const int col = raw_row * 2 + (raw_col > 3 ? 1 : 0);
        const Key& k = kMap[row][col];

        if (k.plain == kShift) {
            if (pressed) {
                shift_held_ = true;
                shift_used_ = false;
            } else {
                shift_held_ = false;
                if (!shift_used_) shift_once_ = !shift_once_;  // tapped alone
            }
            continue;
        }
        if (k.plain == kFn) {
            fn_held_ = pressed;
            continue;
        }
        if (!pressed || k.plain == kNone) continue;

        if (fn_held_) {
            if (k.plain == '`') return 27;     // escape
            if (k.plain == kBack) return kBack;
            continue;
        }
        const bool shifted = shift_held_ || shift_once_;
        if (shift_held_) shift_used_ = true;
        shift_once_ = false;
        const char c = shifted ? k.shifted : k.plain;
        return c == kTab ? ' ' : c;
    }
    return 0;
}
