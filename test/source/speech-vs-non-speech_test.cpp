#include "lib.hpp"

auto main() -> int
{
  auto const lib = library {};

  return lib.name == "speech-vs-non-speech" ? 0 : 1;
}
