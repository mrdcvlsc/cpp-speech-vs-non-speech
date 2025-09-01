# Parent project does not export its library target, so this CML implicitly
# depends on being added from it, i.e. the testing is done only from the build
# tree and is not feasible from an install location

project(speech-vs-non-speechTests LANGUAGES CXX)

# ---- Tests ----

add_executable(speech-vs-non-speech_test source/speech-vs-non-speech_test.cpp)
target_link_libraries(speech-vs-non-speech_test PRIVATE speech-vs-non-speech_lib)
target_compile_features(speech-vs-non-speech_test PRIVATE cxx_std_17)

add_test(NAME speech-vs-non-speech_test COMMAND speech-vs-non-speech_test)

# ---- End-of-file commands ----

add_folders(Test)
