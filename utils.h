#pragma once
#include <cstdint>

enum class Endian {
	Big,
	Small
};


uint32_t swapEndianness(uint32_t value) {
	return (value << 24) | ((value >> 8) & 0x00FF00) |
		((value >> 16) & 0xFF00FF);
}

float reverseByteOrder(float num) {
	uint32_t intVal = (uint32_t)(num);
	intVal = swapEndianness(intVal);
	return (float)(intVal);
}