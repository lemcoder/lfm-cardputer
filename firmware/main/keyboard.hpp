#pragma once
// Cardputer-Adv keyboard: a TCA8418 key-matrix controller on the internal I2C
// bus (address 0x34), polled for its event FIFO. Key layout and the
// controller's row/column remap follow M5Stack's M5Cardputer library.
//
// poll() returns one key at a time: printable ASCII, '\b' (backspace),
// '\n' (enter), 27 (fn + `: escape), or 0 when nothing is pending.
// Aa (shift) applies while held, or to the next key after a tap on its own.
//
// A task reads the controller every 10 ms into a queue, so keys typed while
// the model runs a pass (seconds without a poll() call) aren't lost; the
// controller's own FIFO holds only 10 events.

#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"

class Keyboard {
public:
    bool begin();
    char poll();

private:
    static void task(void* arg);
    char read_hw();

    QueueHandle_t queue_ = nullptr;
    bool shift_held_ = false;
    bool shift_used_ = false;  // a key was typed while shift was held
    bool shift_once_ = false;  // shift tapped alone: applies to the next key
    bool fn_held_ = false;
};
