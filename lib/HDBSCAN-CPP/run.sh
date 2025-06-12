$CXX -c -g -O3 */*.cpp
$CXX -g -mtune-native -march-native -O3 main.cpp *.o
# ./a.out
rm  *.o
