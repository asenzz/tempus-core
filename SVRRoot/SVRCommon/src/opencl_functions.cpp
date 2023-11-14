//#include <stdio.h>
//#include <stdlib.h>
//
//#include <string.h>

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstring>
//#include <stdio.h>

namespace svr {

#include "common/opencl_functions.hpp"

//using namespace std;
//////////////////////////////////////////////////////////////////////////////
//! Find the path for a file assuming that
//! files are found in the searchPath.
//!
//! @return the path if succeeded, otherwise 0
//! @param filename         name of the file
//! @param executable_path  optional absolute path of the executable
//////////////////////////////////////////////////////////////////////////////
char *shrFindFilePath(const char *filename, const char *executable_path)
{
    // <executable_name> defines a variable that is replaced with the name of the executable

    // Typical relative search paths to locate needed companion files (e.g. sample input data, or JIT source files)
    // The origin for the relative search may be the .exe file, a .bat file launching an .exe, a browser .exe launching the .exe or .bat, etc
    const char *searchPath[] =
            {
                    "./"//,                                       // same dir
                    //"./data/",                                  // "/data/" subdir
                    //"./src/",                                   // "/src/" subdir
                    //"./src/<executable_name>/data/",            // "/src/<executable_name>/data/" subdir
                    //"./inc/",                                   // "/inc/" subdir
                    //"../",                                      // up 1 in tree
                    //"../data/",                                 // up 1 in tree, "/data/" subdir
                    //"../src/",                                  // up 1 in tree, "/src/" subdir
                    //"../inc/",                                  // up 1 in tree, "/inc/" subdir
                    //"../OpenCL/src/<executable_name>/",         // up 1 in tree, "/OpenCL/src/<executable_name>/" subdir
                    //"../OpenCL/src/<executable_name>/data/",    // up 1 in tree, "/OpenCL/src/<executable_name>/data/" subdir
                    //"../OpenCL/src/<executable_name>/src/",     // up 1 in tree, "/OpenCL/src/<executable_name>/src/" subdir
                    //"../OpenCL/src/<executable_name>/inc/",     // up 1 in tree, "/OpenCL/src/<executable_name>/inc/" subdir
                    //"../C/src/<executable_name>/",              // up 1 in tree, "/C/src/<executable_name>/" subdir
                    //"../C/src/<executable_name>/data/",         // up 1 in tree, "/C/src/<executable_name>/data/" subdir
                    //"../C/src/<executable_name>/src/",          // up 1 in tree, "/C/src/<executable_name>/src/" subdir
                    //"../C/src/<executable_name>/inc/",          // up 1 in tree, "/C/src/<executable_name>/inc/" subdir
                    //"../DirectCompute/src/<executable_name>/",      // up 1 in tree, "/DirectCompute/src/<executable_name>/" subdir
                    //"../DirectCompute/src/<executable_name>/data/", // up 1 in tree, "/DirectCompute/src/<executable_name>/data/" subdir
                    //"../DirectCompute/src/<executable_name>/src/",  // up 1 in tree, "/DirectCompute/src/<executable_name>/src/" subdir
                    //"../DirectCompute/src/<executable_name>/inc/",  // up 1 in tree, "/DirectCompute/src/<executable_name>/inc/" subdir
                    //"../../",                                   // up 2 in tree
                    //"../../data/",                              // up 2 in tree, "/data/" subdir
                    //"../../src/",                               // up 2 in tree, "/src/" subdir
                    //"../../inc/",                               // up 2 in tree, "/inc/" subdir
                    //"../../../",                                // up 3 in tree
                    //"../../../src/<executable_name>/",          // up 3 in tree, "/src/<executable_name>/" subdir
                    //"../../../src/<executable_name>/data/",     // up 3 in tree, "/src/<executable_name>/data/" subdir
                    //"../../../src/<executable_name>/src/",      // up 3 in tree, "/src/<executable_name>/src/" subdir
                    //"../../../src/<executable_name>/inc/",      // up 3 in tree, "/src/<executable_name>/inc/" subdir
                    //"../../../sandbox/<executable_name>/",      // up 3 in tree, "/sandbox/<executable_name>/" subdir
                    //"../../../sandbox/<executable_name>/data/", // up 3 in tree, "/sandbox/<executable_name>/data/" subdir
                    //"../../../sandbox/<executable_name>/src/",  // up 3 in tree, "/sandbox/<executable_name>/src/" subdir
                    //"../../../sandbox/<executable_name>/inc/"   // up 3 in tree, "/sandbox/<executable_name>/inc/" subdir
            };

    // Extract the executable name
    std::string executable_name;
    if (executable_path != 0) {
        executable_name = std::string(executable_path);

        #ifdef _WIN32
        // Windows path delimiter
        size_t delimiter_pos = executable_name.find_last_of('\\');
        executable_name.erase(0, delimiter_pos + 1);

        if (executable_name.rfind(".exe") != string::npos)
        {
            // we strip .exe, only if the .exe is found
            executable_name.resize(executable_name.size() - 4);
        }
        #else
        // Linux & OSX path delimiter
        size_t delimiter_pos = executable_name.find_last_of('/');
        executable_name.erase(0, delimiter_pos + 1);
        #endif

    }

    // Loop over all search paths and return the first hit
    for (unsigned int i = 0; i < sizeof(searchPath) / sizeof(char *); ++i) {
        std::string path(searchPath[i]);
        size_t executable_name_pos = path.find("<executable_name>");

        // If there is executable_name variable in the searchPath
        // replace it with the value
        if (executable_name_pos != std::string::npos) {
            if (executable_path != 0) {
                path.replace(executable_name_pos, std::strlen("<executable_name>"), executable_name);

            } else {
                // Skip this path entry if no executable argument is given
                continue;
            }
        }

        // Test if the file exists
        path.append(filename);
        std::fstream fh(path.c_str(), std::fstream::in);
        if (fh.good()) {
            // File found
            // returning an allocated array here for backwards compatibility reasons
            char *file_path = (char *) malloc(path.length() + 1);
            #ifdef _WIN32
            strcpy_s(file_path, path.length() + 1, path.c_str());
            #else
            strcpy(file_path, path.c_str());
            #endif
            return file_path;
        }
    }

    // File not found
    return 0;
}


//////////////////////////////////////////////////////////////////////////////
//! Loads a Program file and prepends the cPreamble to the code.
//!
//! @return the source string if succeeded, 0 otherwise
//! @param cFilename        program filename
//! @param cPreamble        code that is prepended to the loaded file, typically a set of #defines or a header
//! @param szFinalLength    returned length of the code string
//////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////
//! Loads a Program file and prepends the cPreamble to the code.
//!
//! @return the source string if succeeded, 0 otherwise
//! @param cFilename        program filename
//! @param cPreamble        code that is prepended to the loaded file, typically a set of #defines or a header
//! @param szFinalLength    returned length of the code string
//////////////////////////////////////////////////////////////////////////////
char *oclLoadProgSource(const char *cFilename, const char *cPreamble, size_t *szFinalLength)
{
    // locals
    FILE *pFileStream = NULL;
    size_t szSourceLength;

    // open the OpenCL source code file
    #ifdef _WIN32   // Windows version
    if(fopen_s(&pFileStream, cFilename, "rb") != 0)
        {
            return NULL;
        }
    #else           // Linux version
    pFileStream = fopen(cFilename, "rb");
    if (pFileStream == 0) {
        return NULL;
    }
    #endif

    size_t szPreambleLength = strlen(cPreamble);

    // get the length of the source code
    fseek(pFileStream, 0, SEEK_END);
    szSourceLength = ftell(pFileStream);
    fseek(pFileStream, 0, SEEK_SET);

    // allocate a buffer for the source code string and read it in
    char *cSourceString = (char *) malloc(szSourceLength + szPreambleLength + 1);
    memcpy(cSourceString, cPreamble, szPreambleLength);
    if (fread((cSourceString) + szPreambleLength, szSourceLength, 1, pFileStream) != 1) {
        fclose(pFileStream);
        free(cSourceString);
        return 0;
    }

    // close the file and return the total length of the combined (preamble + source) string
    fclose(pFileStream);
    if (szFinalLength != 0) {
        *szFinalLength = szSourceLength + szPreambleLength;
    }
    cSourceString[szSourceLength + szPreambleLength] = '\0';

    return cSourceString;
}


//////////////////////////////////////////////////////////////////////////////
//! Loads a Program file and prepends the cPreamble to the code.
//!
//! @return the source string if succeeded, 0 otherwise
//! @param cFilename        program filename
//! @param cPreamble        code that is prepended to the loaded file, typically a set of #defines or a header
//! @param szFinalLength    returned length of the code string
//////////////////////////////////////////////////////////////////////////////
char *oclLoadProgSource_local(const char *cFilename, const char *cPreamble, size_t *szFinalLength)
{
    // locals
    FILE *pFileStream = NULL;
    size_t szSourceLength;

    // open the OpenCL source code file
    #ifdef _WIN32   // Windows version
    if(fopen_s(&pFileStream, cFilename, "rb") != 0)
        {
            return NULL;
        }
    #else           // Linux version
    pFileStream = fopen(cFilename, "rb");
    if (pFileStream == 0) {
        return NULL;
    }
    #endif

    size_t szPreambleLength = strlen(cPreamble);

    // get the length of the source code
    fseek(pFileStream, 0, SEEK_END);
    szSourceLength = ftell(pFileStream);
    fseek(pFileStream, 0, SEEK_SET);

    // allocate a buffer for the source code string and read it in
    char *cSourceString = (char *) malloc(szSourceLength + szPreambleLength + 1);
    memcpy(cSourceString, cPreamble, szPreambleLength);
    if (fread((cSourceString) + szPreambleLength, szSourceLength, 1, pFileStream) != 1) {
        fclose(pFileStream);
        free(cSourceString);
        return 0;
    }

    // close the file and return the total length of the combined (preamble + source) string
    fclose(pFileStream);
    if (szFinalLength != 0) {
        *szFinalLength = szSourceLength + szPreambleLength;
    }
    cSourceString[szSourceLength + szPreambleLength] = '\0';

    return cSourceString;
}



}


