SOURCES=$(shell find . -name "*.cpp")
CXXFLAGS?=-std=c++11 -Wall -mtune=native -O3 -march=native
OBJECTS=$(SOURCES:%.cpp=%.o)
TARGET=libhdbscan-cpp.so

.PHONY: all
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(LINK.cpp) $^ -std=c++11 $(LOADLIBES) $(LDLIBS) -fPIC -shared -o $@

.PHONY: clean
clean:
	rm -f $(OBJECTS) $(TARGET)
