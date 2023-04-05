#pragma once

#include "Platform.h"

namespace bcm3 {

// Obtained from https://rigtorp.se/spinlock/
struct spinlock {
	std::atomic<bool> lock_ = {0};

	void lock() noexcept {
		for (;;) {
			// Optimistically assume the lock is free on the first try
			if (!lock_.exchange(true, std::memory_order_acquire)) {
				return;
			}
			// Wait for lock to be released without generating cache misses
			while (lock_.load(std::memory_order_relaxed)) {
				// Issue X86 PAUSE or ARM YIELD instruction to reduce contention between hyper-threads
#if defined(__arm__)
				__asm__ __volatile__("yield");
#else
#if PLATFORM_LINUX
				__builtin_ia32_pause();
#elif PLATFORM_WINDOWS
				_mm_pause();
#else
#error
#endif
#endif
			}
		}
	}

	bool try_lock() noexcept {
		// First do a relaxed load to check if lock is free in order to prevent
		// unnecessary cache misses if someone does while(!try_lock())
		return !lock_.load(std::memory_order_relaxed) && !lock_.exchange(true, std::memory_order_acquire);
	}

	void unlock() noexcept {
		lock_.store(false, std::memory_order_release);
	}
};

}
