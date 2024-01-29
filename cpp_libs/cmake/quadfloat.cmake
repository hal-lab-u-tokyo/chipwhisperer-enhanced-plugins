
set(SOURCE_CODE
"#include <iostream>\n#include <climits>\nint main() {\nstd::cout << (CHAR_BIT * sizeof(long double)) << std::endl;\nreturn 0;\n}\n"
)

file(WRITE ${CMAKE_BINARY_DIR}/check_longdouble_size.cpp "${SOURCE_CODE}")

# run test program to determine the size of long double
try_run(RUN_RESULT_VAR COMPILE_RESULT_VAR
	${CMAKE_BINARY_DIR}/bin
	${CMAKE_BINARY_DIR}/check_longdouble_size.cpp
	RUN_OUTPUT_VARIABLE LONG_DOBULE_SIZE
)

# check result
if(COMPILE_RESULT_VAR AND RUN_RESULT_VAR EQUAL 0)
	if (LONG_DOBULE_SIZE EQUAL 128)
		set(HAVE_QUAD_PRECISION TRUE)
		message(STATUS "Quad-precision floating point is available")
	else()
		set(HAVE_QUAD_PRECISION FALSE)
		message(STATUS "Quad-precision floating point is not available. Software emulation will be used")
	endif()
else()
	message(STATUS "Failed to determine the size of long double")
endif()
