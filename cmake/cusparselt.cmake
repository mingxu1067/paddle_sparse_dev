if(NOT WITH_GPU)
    return()
endif()

if(WIN32)
    return()
else()
    set(CUSPARSELT_ROOT "/usr" CACHE PATH "CUSPARSELT ROOT")
    set(CUSPARSELT_LIB libcusparseLt_static.a)
    set(CUSPARSELT_RT libcusparseLt.so)
endif()

find_path(CUSPARSELT_INCLUDE_DIR cusparseLt.h
    PATHS ${CUSPARSELT_ROOT} ${CUSPARSELT_ROOT}/include
    $ENV{CUSPARSELT_ROOT} $ENV{CUSPARSELT_ROOT}/include
    NO_DEFAULT_PATH
)

find_path(CUSPARSELT_LIBRARY_DIR NAMES ${CUSPARSELT_LIB} ${CUSPARSELT_RT}
    PATHS ${CUSPARSELT_ROOT} ${CUSPARSELT_ROOT}/lib
    $ENV{CUSPARSELT_ROOT} $ENV{CUSPARSELT_ROOT}/lib
    NO_DEFAULT_PATH
    DOC "Path to cuSparseLt library."
)

find_library(CUSPARSELT_LIBRARY NAMES ${CUSPARSELT_LIB} ${CUSPARSELT_RT}
    PATHS ${CUSPARSELT_LIBRARY_DIR}
    NO_DEFAULT_PATH
    DOC "Path to cuSparseLt library.")

if(CUSPARSELT_INCLUDE_DIR AND CUSPARSELT_LIBRARY)
    set(CUSPARSELT_FOUND ON)
else()
    set(CUSPARSELT_FOUND OFF)
    message(WARNING "cuSparseLt is disabled.")
endif()

if(CUSPARSELT_FOUND)
    include_directories(${CUSPARSELT_INCLUDE_DIR})
    link_directories(${CUSPARSELT_LIBRARY})
    add_definitions(-DPADDLE_WITH_CUSPARSELT)
endif()
