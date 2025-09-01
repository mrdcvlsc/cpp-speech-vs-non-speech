.PHONY: default config build release debug clean

default:
	@echo make config - configure with cmake
	@echo make build - compile program

config:
	cmake -B build -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DBUILD_SHARED_LIBS=FALSE -DCMAKE_BUILD_TYPE=Release

build:
	cmake --build build --config Release

release:
	cmake -B build -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DBUILD_SHARED_LIBS=FALSE -DCMAKE_BUILD_TYPE=Release
	cmake --build build --config Release -j3

debug:
	cmake -B build -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DBUILD_SHARED_LIBS=FALSE -DCMAKE_BUILD_TYPE=Debug
	cmake --build build --config Debug -j3

clean:
	rm -rf build external
