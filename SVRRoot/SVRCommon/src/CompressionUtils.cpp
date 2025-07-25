/*
 * CompressionUtils.cpp
 *
 *  Created on: April 8, 2015
 *      Author: skondrat
 */

#include "util/CompressionUtils.hpp"

// #pragma warning ( push )
// #pragma warning (disable: "-Wunsupported-floating-point-opt")


namespace svr
{
namespace common
{

static const std::string base64_chars =
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
             "abcdefghijklmnopqrstuvwxyz"
             "0123456789+/";

static const char divider = '%';

static inline bool is_base64(unsigned char c)
{
  return (isalnum(c) || (c == '+') || (c == '/'));
}

std::string compress(const char * input, int size)
{
    char* p_compressed = new char[size];
    int compressedSize = LZ4_compress_default(input, p_compressed, size, size);
    std::string compressedString(p_compressed, p_compressed + compressedSize);
    delete[] p_compressed;
    compressedString = std::to_string(size) + divider + compressedString;
    return compressedString;
}

std::string decompress(const char * input, int size)
{
    const char * dividerPos = std::find(input, input + size, divider);
    std::string compressSizeStr = std::string(input, dividerPos); /* TODO Verify string initialization is correct */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
    int compressSize = std::stoi(compressSizeStr);
    char* pcharDecompressed = new char[compressSize];
#pragma GCC diagnostic pop

    int decompressSize = LZ4_decompress_safe(dividerPos + 1, pcharDecompressed, size - compressSizeStr.size() - 1,
                                             compressSize);
    std::string decompressedString(pcharDecompressed, pcharDecompressed + decompressSize);
    delete[] pcharDecompressed;
    return decompressedString;
}

std::string base64_encode(unsigned char const *bytes_to_encode, unsigned in_len)
{
  std::string ret;
  int i = 0;
  unsigned char char_array_3[3];
  unsigned char char_array_4[4];

  while (in_len--) {
    char_array_3[i++] = *(bytes_to_encode++);
    if (i == 3) {
      char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
      char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
      char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
      char_array_4[3] = char_array_3[2] & 0x3f;

      for(i = 0; (i <4) ; i++)
        ret += base64_chars[char_array_4[i]];
      i = 0;
    }
  }

    if (i) {
    int j = 0;
    for(j = i; j < 3; j++)
      char_array_3[j] = '\0';

    char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
    char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
    char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
    char_array_4[3] = char_array_3[2] & 0x3f;

    for (j = 0; (j < i + 1); j++)
      ret += base64_chars[char_array_4[j]];

    while((i++ < 3))
      ret += '=';

  }

  return ret;

}

std::string base64_decode(const std::string &encoded_string)
{
  int in_len = encoded_string.size();
  int i = 0;
  int in_ = 0;
  unsigned char char_array_4[4], char_array_3[3];
  std::string ret;

  while (in_len-- && ( encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
        char_array_4[i++] = encoded_string[in_];
        in_++;
    if (i ==4) {
      for (i = 0; i <4; i++)
        char_array_4[i] = base64_chars.find(char_array_4[i]);

      char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
      char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
      char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

      for (i = 0; (i < 3); i++)
        ret += char_array_3[i];
      i = 0;
    }
  }

  if (i) {
    int j = 0;
    for (j = i; j <4; j++)
      char_array_4[j] = 0;

    for (j = 0; j <4; j++)
      char_array_4[j] = base64_chars.find(char_array_4[j]);

    char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
    char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
    char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

    for (j = 0; (j < i - 1); j++) ret += char_array_3[j];
  }

  return ret;
}


}
}

