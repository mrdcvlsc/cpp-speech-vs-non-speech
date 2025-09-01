install(
    TARGETS speech-vs-non-speech_exe
    RUNTIME COMPONENT speech-vs-non-speech_Runtime
)

if(PROJECT_IS_TOP_LEVEL)
  include(CPack)
endif()
