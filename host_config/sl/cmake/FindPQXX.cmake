set( PQXX_FOUND 1 CACHE INTERNAL "PQXX found" FORCE )
set( PQXX_INCLUDE_DIRECTORIES "$ENV{TEMPUS_LIBRARY_DIR}/libpqxx-5.0.1/include" CACHE STRING "Include directories for PostGreSQL C++ library" FORCE )
set( PQXX_LIBRARIES "pqxx" CACHE STRING "Link libraries for PostGreSQL C++ interface" FORCE )
set( PQXX_LIBRARIES_DIRECTORY "$ENV{TEMPUS_LIBRARY_DIR}/libpqxx-5.0.1/lib" CACHE STRING "Library directories" FORCE )
