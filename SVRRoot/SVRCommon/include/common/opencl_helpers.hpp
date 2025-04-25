#ifdef ENABLE_OPENCL


#ifndef OPENCL_FUNCTIONS_HPP
#define OPENCL_FUNCTIONS_HPP

#include <CL/cl.h>

//////////////////////////////////////////////////////////////////////////////
//! Find the path for a file assuming that
//! files are found in the searchPath.
//!
//! @return the path if succeeded, otherwise 0
//! @param filename         name of the file
//! @param executable_path  optional absolute path of the executable
//////////////////////////////////////////////////////////////////////////////
char* shrFindFilePath(const char* filename, const char* executable_path) ;
//////////////////////////////////////////////////////////////////////////////
//! Loads a Program file and prepends the cPreamble to the code.
//!
//! @return the source string if succeeded, 0 otherwise
//! @param cFilename        program filename
//! @param cPreamble        code that is prepended to the loaded file, typically a set of #defines or a header
//! @param szFinalLength    returned length of the code string
//////////////////////////////////////////////////////////////////////////////
char* oclLoadProgSource(const char* cFilename, const char* cPreamble, size_t* szFinalLength);
char* oclLoadProgSource_local(const char* cFilename, const char* cPreamble, size_t* szFinalLength);

#endif
//End of OpenCL_FUNCTIONS_H

#endif
